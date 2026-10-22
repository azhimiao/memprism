"""Offline checks for deploy/server public-host packaging (no network)."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "deploy" / "server"
COMPOSE_BASE = ROOT / "deploy" / "compose.yaml"
COMPOSE_SERVER = SERVER / "compose.server.yaml"
ENV_EXAMPLE = SERVER / "env.server.example"
NGINX_CONF = SERVER / "nginx-test.memprism.com.conf"
SHELL_SCRIPTS = [
    SERVER / "bootstrap-ubuntu.sh",
    SERVER / "deploy.sh",
    SERVER / "backup.sh",
    SERVER / "configure-and-deploy.sh",
    SERVER / "status.sh",
    SERVER / "setup-https.sh",
]
STREAMED_SCRIPTS = ["configure-and-deploy.sh", "status.sh", "setup-https.sh"]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"),
]

def _find_bash() -> Path | None:
    for candidate in (
        Path("/bin/bash"),
        Path("/usr/bin/bash"),
        Path(r"C:\Program Files\Git\bin\bash.exe"),
    ):
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_bash_n(script: Path) -> None:
    bash = _find_bash()
    if bash is None:
        pytest.skip("bash not found (skipped bash -n only)")
    proc = subprocess.run(
        [str(bash), "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _load_compose_server_ports() -> list[str]:
    """Minimal YAML subset parser for compose.server.yaml ports list."""
    text = COMPOSE_SERVER.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_ports = False
    ports: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ports:"):
            in_ports = True
            continue
        if in_ports:
            if stripped.startswith("- "):
                ports.append(stripped[2:].strip().strip('"').strip("'"))
                continue
            if stripped and not stripped.startswith("#"):
                break
    assert ports, "ports list not found in compose.server.yaml"
    return ports


def test_compose_server_yaml_loopback_host_port_replaces_base_ports() -> None:
    ports = _load_compose_server_ports()
    assert ports == ["127.0.0.1:28717:8787"]
    text = COMPOSE_SERVER.read_text(encoding="utf-8")
    assert "ports: !override" in text and 'AML_PORT: "8787"' in text
    assert 'HEALTH_URL="http://127.0.0.1:28717/health"' in (SERVER / "deploy.sh").read_text(encoding="utf-8")
    nginx = NGINX_CONF.read_text(encoding="utf-8")
    assert nginx.count("proxy_pass http://127.0.0.1:28717;") == 3 and "8787" not in nginx


def test_acme_only_site_serves_challenge_without_certificates() -> None:
    text = (SERVER / "nginx-test.memprism.com.acme.conf").read_text(encoding="utf-8")
    assert "server_name test.memprism.com;" in text and "location /.well-known/acme-challenge/" in text
    assert "listen 443" not in text and "ssl_certificate" not in text and "proxy_pass" not in text


def test_bootstrap_reuses_existing_nginx_without_upgrades() -> None:
    text = (SERVER / "bootstrap-ubuntu.sh").read_text(encoding="utf-8")
    assert "apt-get install -y -qq --no-upgrade" in text
    base = text.split("packages=(", 1)[1].split(")", 1)[0].split()
    assert "nginx" not in base and "python3-certbot-nginx" not in text
    assert "if ! command -v nginx >/dev/null 2>&1; then\n  packages+=(nginx)" in text


def test_env_server_example_hybrid_guardrails() -> None:
    values = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    assert values["AML_COORDINATOR_LOCK_TIMEOUT_SECONDS"] == "30"
    assert values["AML_MAX_CONCURRENT_REQUESTS"] == "32"


def test_env_server_example_no_nonempty_secrets() -> None:
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for key in ("AML_SERVICE_API_KEY", "DASHSCOPE_API_KEY"):
            if line.startswith(f"{key}="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                assert value == "", f"{key} must be empty in example"


def test_nginx_whitelist_and_redirect() -> None:
    text = NGINX_CONF.read_text(encoding="utf-8")
    assert "return 301 https://" in text
    for path in ("/health", "/add", "/search"):
        assert f"location = {path}" in text or f"location = {path} " in text
    http_block, https_block = text.split("listen 443 ssl;", 1)
    assert "location /.well-known/acme-challenge/" in http_block
    assert http_block.index("acme-challenge") < http_block.index("return 301 https://")
    for path in ("/add", "/search"):
        block = https_block.split(f"location = {path}", 1)[1].split("}", 1)[0]
        assert "proxy_read_timeout 1800s;" in block and "proxy_send_timeout 1800s;" in block


def test_scripts_contain_no_hardcoded_secrets() -> None:
    combined = "\n".join(p.read_text(encoding="utf-8") for p in SHELL_SCRIPTS)
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(combined), pattern.pattern


def test_bootstrap_keeps_existing_firewall_rules() -> None:
    text = (SERVER / "bootstrap-ubuntu.sh").read_text(encoding="utf-8")
    assert "ufw --force reset" not in text and "ufw default" not in text
    enable = text.index("ufw --force enable")
    assert 'if [[ "${AML_ENABLE_UFW:-0}" == "1" ]]; then' in text[:enable]


def test_streamed_configure_script_keeps_keys_out_of_output_and_arguments() -> None:
    raw = (SERVER / "configure-and-deploy.sh").read_bytes()
    assert b"\r" not in raw and raw.isascii()
    text = raw.decode()
    assert text.rstrip().endswith('main "$@" </dev/null; exit $?')
    for line in text.splitlines():
        if re.search(r"\b(echo|printf)\b", line) and re.search(r"dashscope_key|service_key", line):
            assert line.strip().startswith("AML_SERVICE_API_KEY=") or "printf 'AML_SERVICE_API_KEY=%s" in line \
                or "printf 'DASHSCOPE_API_KEY=%s" in line or "printf 'header = \"Authorization: Bearer %s" in line, line
    assert "| curl -s -o /dev/null -w '%{http_code}' -K -" in text
    assert "keeping the existing service key" in text
    wrapper = (SERVER / "remote-configure.ps1").read_text(encoding="utf-8")
    assert "Invoke-AmlWithDashScopeKey" in wrapper and "ssh -T $Target 'bash -s'" in wrapper
    assert "Write-Host" not in wrapper and "Out-File" not in wrapper and "Set-Content" not in wrapper


@pytest.mark.parametrize("name", STREAMED_SCRIPTS)
def test_streamed_scripts_run_from_stdin_safely(name: str) -> None:
    raw = (SERVER / name).read_bytes()
    assert b"\r" not in raw and raw.isascii()
    text = raw.decode()
    last = text.rstrip().splitlines()[-1]
    assert last.startswith("main ") and "</dev/null; exit " in last
    assert not re.search(r"echo[^\n]*\$\{?(dashscope_key|service_key)", text)


def test_https_setup_only_reloads_after_config_test() -> None:
    text = (SERVER / "setup-https.sh").read_text(encoding="utf-8")
    apply = text.split("apply_site() {", 1)[1].split("\n  }\n", 1)[0]
    assert apply.index("if ! nginx -t; then") < apply.index("systemctl reload nginx")
    assert 'rm -f "${link}"' in apply
    assert '--cert-name "${domain}"' in text and "certbot renew --dry-run\n" not in text
    assert 'if [[ "${resolved}" != "${expected_ip}" ]]; then' in text


def test_bootstrap_data_dir_is_writable_by_container_user() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    uid = re.search(r"useradd --uid (\d+) --gid (\d+)", dockerfile)
    assert uid is not None
    bootstrap = (SERVER / "bootstrap-ubuntu.sh").read_text(encoding="utf-8")
    assert f"install -d -m 0750 -o {uid.group(1)} -g {uid.group(2)} /var/lib/aml" in bootstrap
    assert "/var/lib/aml:/var/lib/aml" in COMPOSE_SERVER.read_text(encoding="utf-8")


def _make_release():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_release", SERVER / "make_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_origin(release) -> dict:
    return {"revision": "test", "new": {p.name: release.sha256(p.read_bytes()) for p in (ROOT / "app").glob("*.py")}}


def test_release_archive_has_deploy_layout_and_manifest(tmp_path: Path) -> None:
    import json
    import tarfile
    release = _make_release()
    origin = tmp_path / "origin.json"
    origin.write_text(json.dumps(_current_origin(release)), encoding="utf-8")
    result = release.build(tmp_path / "release.tar.gz", origin)
    with tarfile.open(result["out"]) as tar:
        names = set(tar.getnames())
        manifest = json.loads(tar.extractfile("our-system/RELEASE.json").read())
        modes = {m.name: m.mode for m in tar.getmembers()}
    for required in ("our-system/deploy/compose.yaml", "our-system/deploy/server/compose.server.yaml",
                     "our-system/deploy/Dockerfile", "our-system/requirements-cycle2.txt", "our-system/app/api.py"):
        assert required in names
    assert not any("__pycache__" in n or n.endswith((".pyc", ".db", ".sqlite")) or "/pytest-tmp" in n
                   or n.startswith(("our-system/tests", "our-system/deploy/local/")) for n in names)
    assert modes["our-system/deploy/server/deploy.sh"] == 0o755
    assert set(manifest["files"]) == {n.removeprefix("our-system/") for n in names} - {"RELEASE.json"}
    with pytest.raises(SystemExit, match="exists"):
        release.build(tmp_path / "release.tar.gz", origin)


def test_release_rejects_unregistered_sources_and_crlf_scripts(tmp_path: Path) -> None:
    import json
    release = _make_release()
    stale = _current_origin(release)
    stale["new"]["api.py"] = "0" * 64
    origin = tmp_path / "stale.json"
    origin.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not match candidate origin: \\['api.py'\\]"):
        release.build(tmp_path / "stale.tar.gz", origin)
    assert not (tmp_path / "stale.tar.gz").exists()

    base = tmp_path / "base"
    (base / "app").mkdir(parents=True)
    (base / "deploy").mkdir()
    (base / "app" / "x.py").write_bytes(b"x = 1\n")
    (base / "deploy" / "run.sh").write_bytes(b"#!/usr/bin/env bash\r\necho ok\r\n")
    (base / "requirements-cycle2.txt").write_bytes(b"")
    files = release.release_files(base)
    with pytest.raises(SystemExit, match="CRLF"):
        release.check_sources(files, {"new": {"x.py": release.sha256(b"x = 1\n")}}, base)


def test_compose_merge_config_if_docker_cli() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not installed")
    env_file = ENV_EXAMPLE
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_BASE),
            "-f",
            str(COMPOSE_SERVER),
            "config",
        ],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "AML_ENV_FILE": str(env_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    out = proc.stdout
    assert "8787" in out and ("127.0.0.1" in out or "host_ip: 127.0.0.1" in out)

"""Offline checks for deploy/local PowerShell scripts and cloudflared template."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "deploy" / "local"
SERVICE_PS1 = LOCAL / "aml-service.ps1"
TUNNEL_PS1 = LOCAL / "aml-tunnel.ps1"
TEMPLATE_YML = LOCAL / "cloudflared-config.template.yml"

# Keys from scripts.v4_live_smoke.build_service_env (Hybrid), excluding secret values.
HYBRID_ENV_KEYS = frozenset(
    {
        "AML_PROFILE",
        "AML_RETRIEVAL_MODE",
        "AML_ENABLE_VECTOR",
        "AML_DB_PATH",
        "AML_AUTH_MODE",
        "AML_SERVICE_KEY_ENV",
        "AML_SERVICE_API_KEY",
        "AML_EMBEDDING_PROVIDER",
        "AML_EMBEDDING_ENDPOINT",
        "AML_EMBEDDING_REGION",
        "AML_EMBEDDING_MODEL",
        "AML_EMBEDDING_DIMENSIONS",
        "AML_EMBEDDING_REVISION",
        "AML_EMBEDDING_KEY_ENV",
        "AML_EMBEDDING_BATCH_SIZE",
        "AML_EMBEDDING_TIMEOUT_SECONDS",
        "AML_EMBEDDING_MAX_RETRIES",
        "AML_EMBEDDING_MAX_INPUT_BYTES",
        "AML_EMBEDDING_MIN_INTERVAL_SECONDS",
        "AML_COORDINATOR_LOCK_TIMEOUT_SECONDS",
        "AML_MAX_CONCURRENT_REQUESTS",
        "AML_MAX_REQUEST_BYTES",
        "AML_CANDIDATE_LIMIT",
        "AML_LEXICAL_QUOTA",
        "AML_DENSE_QUOTA",
        "AML_MAX_RETURN_ITEMS",
        "AML_RETURN_CHAR_BUDGET",
        "AML_RETURN_TOKEN_BUDGET",
        "AML_PACKING_POLICY",
        "AML_ROUTE_POLICY",
        "AML_ENABLE_GOVERNANCE",
        "AML_EXACT_DENSE_ACCELERATION",
        "AML_HOST",
        "AML_PORT",
        "TIKTOKEN_CACHE_DIR",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUTF8",
        "DASHSCOPE_API_KEY",
    }
)


def _parse_ps1(path: Path) -> list:
    cmd = (
        "$e=$null; $t=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path.as_posix()}', [ref]$t, [ref]$e); "
        "if ($e) { $e | ForEach-Object { $_.ToString() }; exit 1 } else { exit 0 }"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc


@pytest.mark.parametrize("script", [SERVICE_PS1, TUNNEL_PS1])
def test_powershell_parse_no_syntax_errors(script: Path) -> None:
    proc = _parse_ps1(script)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_service_script_static_constraints() -> None:
    text = SERVICE_PS1.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert re.search(r"\b8787\b", text)
    assert "aml-hybrid.sqlite" in text
    assert "aml-fts.sqlite" in text

    # Must not emit secrets via Write-Output / Write-Host
    forbidden = re.findall(
        r"Write-(Output|Host)\s+(\$\w*(?:Key|plain|Plain|amlPlain)[^\r\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    assert not forbidden, f"possible key leak: {forbidden}"

    env_assigns = set(
        re.findall(
            r"\$env:((?:AML_[A-Z0-9_]+)|PYTHONDONTWRITEBYTECODE|PYTHONUTF8|TIKTOKEN_CACHE_DIR|DASHSCOPE_API_KEY)\s*=",
            text,
        )
    )
    env_assigns.add("AML_SERVICE_API_KEY")  # set inside Invoke-AmlWithServiceKey
    missing = HYBRID_ENV_KEYS - env_assigns
    assert not missing, f"missing Hybrid env keys in script: {sorted(missing)}"


def test_tunnel_template_yaml_or_text() -> None:
    raw = TEMPLATE_YML.read_text(encoding="utf-8")
    assert "http_status:404" in raw.replace(" ", "")
    assert "{{TUNNEL_UUID}}" in raw
    assert "127.0.0.1:8787" in raw
    try:
        import yaml  # type: ignore
    except ImportError:
        assert "ingress:" in raw
        assert "service:" in raw
        return
    data = yaml.safe_load(raw.replace("{{TUNNEL_UUID}}", "00000000-0000-0000-0000-000000000001")
                           .replace("{{CREDENTIALS_FILE}}", "C:/Users/x/.cloudflared/t.json")
                           .replace("{{HOSTNAME}}", "example.com"))
    ingress = data.get("ingress") or []
    assert ingress
    assert any(
        isinstance(rule, dict) and rule.get("service") == "http_status:404"
        for rule in ingress
    )


def test_fts_not_supported_marker() -> None:
    text = SERVICE_PS1.read_text(encoding="utf-8")
    assert "not_supported" in text
    assert "Fts" in text

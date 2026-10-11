"""Build the server tarball consumed by deploy.sh (`/opt/aml/our-system/...` layout).

Only the Docker build inputs are packed. App sources must match a registered candidate origin.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

OUR_SYSTEM = Path(__file__).resolve().parents[2]
INCLUDE = ("requirements-cycle2.txt", "app", "deploy")
EXCLUDED_PARTS = {"__pycache__"}
EXCLUDED_PART_PREFIXES = ("pytest-tmp",)
EXCLUDED_DIRS = (("deploy", "local"),)
EXCLUDED_NAMES = {".env"}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".sqlite", ".sqlite-wal", ".sqlite-shm", ".dpapi"}
SECRET_PATTERNS = (re.compile(rb"sk-[A-Za-z0-9]{20,}"), re.compile(rb"Bearer\s+[A-Za-z0-9._\-]{20,}"))
FIXED_MTIME = 0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def release_files(base: Path = OUR_SYSTEM) -> list[Path]:
    files: list[Path] = []
    for entry in INCLUDE:
        path = base / entry
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for p in candidates:
            rel = p.relative_to(base)
            if (EXCLUDED_PARTS & set(rel.parts) or any(part.startswith(EXCLUDED_PART_PREFIXES) for part in rel.parts)
                    or any(rel.parts[:len(d)] == d for d in EXCLUDED_DIRS)
                    or p.name in EXCLUDED_NAMES or p.suffix in EXCLUDED_SUFFIXES):
                continue
            files.append(p)
    return files


def check_sources(files: list[Path], origin: dict, base: Path = OUR_SYSTEM) -> None:
    app = {p.name: sha256(p.read_bytes()) for p in files if p.parent == base / "app" and p.suffix == ".py"}
    if app != origin["new"]:
        changed = sorted(set(app) ^ set(origin["new"]) | {n for n in set(app) & set(origin["new"]) if app[n] != origin["new"][n]})
        raise SystemExit(f"app sources do not match candidate origin: {changed}")
    for p in files:
        data = p.read_bytes()
        if p.suffix == ".sh" and b"\r\n" in data:
            raise SystemExit(f"CRLF line endings in {p.relative_to(base)}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise SystemExit(f"secret-like content in {p.relative_to(base)}")


def build(out: Path, origin_path: Path, base: Path = OUR_SYSTEM) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new output path")
    origin_bytes = origin_path.read_bytes()
    origin = json.loads(origin_bytes)
    files = release_files(base)
    check_sources(files, origin, base)
    manifest = {
        "revision": origin["revision"],
        "candidate_origin_sha256": sha256(origin_bytes),
        "files": {p.relative_to(base).as_posix(): sha256(p.read_bytes()) for p in files},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "x:gz", format=tarfile.PAX_FORMAT) as tar:
        entries = [(f"our-system/{rel}", (base / rel).read_bytes()) for rel in manifest["files"]]
        entries.append(("our-system/RELEASE.json", manifest_bytes))
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(data), FIXED_MTIME, 0, 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    return {"out": str(out), "sha256": sha256(out.read_bytes()), "files": len(manifest["files"]),
            "revision": manifest["revision"], "candidate_origin_sha256": manifest["candidate_origin_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_release")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True, help="candidate-origin.json the app sources must match")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out, args.origin)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Start Add/Search with atomic source-span publication by default."""

from __future__ import annotations

import os
import logging
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import create_app
from app.config import Settings


def service_settings() -> Settings:
    settings = Settings.from_env()
    if "AML_PROFILE" not in os.environ:
        profile = "span_dense_sync" if settings.vector_enabled or settings.embedding is not None else "span_sync"
        settings = replace(settings, profile_id=profile)
    settings.ensure_phase_a_defaults()
    return settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = service_settings()
    app = create_app(settings)
    host = os.environ.get("AML_HOST", "127.0.0.1")
    if host not in {"127.0.0.1","localhost","::1"} and settings.auth_mode == "none":
        raise RuntimeError("Non-loopback binding requires an explicitly configured API authentication mode")
    port = int(os.environ.get("AML_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

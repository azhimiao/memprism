"""Check configuration without opening a DB or making an embedding request."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw_fts", "raw_dense", "raw_hybrid_rrf"], default="raw_dense")
    args = parser.parse_args()
    os.environ["AML_RETRIEVAL_MODE"] = args.mode  # only this short-lived process
    required = ["AML_EMBEDDING_PROVIDER", "AML_EMBEDDING_ENDPOINT", "AML_EMBEDDING_DIMENSIONS"]
    missing = [name for name in required if not os.environ.get(name)] if args.mode != "raw_fts" else []
    key_name = os.environ.get("AML_EMBEDDING_KEY_ENV", "AML_EMBEDDING_API_KEY")
    if args.mode != "raw_fts" and not os.environ.get(key_name):
        missing.append("embedding_api_key_in_configured_environment_variable")
    try:
        settings = Settings.from_env()
        settings.ensure_phase_a_defaults()
    except (ValueError, RuntimeError, TypeError):
        print(json.dumps({"configuration_ready": False, "mode": args.mode,
                          "reason": "invalid_configuration", "missing": missing,
                          "model_called": False, "scope": "configuration_only"}))
        return 2
    ready = not missing
    print(json.dumps({"configuration_ready": ready, "mode": args.mode, "missing": missing,
                      "model_called": False, "scope": "configuration_only_not_service_or_quality_validation",
                      "candidate_limit": settings.candidate_limit,
                      "return_char_budget": settings.return_char_budget}, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

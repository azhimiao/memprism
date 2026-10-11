"""Unauthenticated loopback readiness; HTTP 200 alone is insufficient."""
import json
import os
import sys
from urllib.request import build_opener, ProxyHandler


def healthy(payload):
    return (isinstance(payload, dict) and payload.get("status") == "ok"
            and payload.get("control_ready") is True and payload.get("data_route_ready") is True)


def main():
    try:
        port = int(os.environ.get("AML_PORT", "8000"))
        if not 1 <= port <= 65535:
            return 1
        with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return 0 if response.status == 200 and healthy(json.load(response)) else 1
    except (OSError, ValueError, TypeError):
        return 1


if __name__ == "__main__":
    sys.exit(main())

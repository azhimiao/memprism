# Evaluation transport reference components

These three modules are byte-exact copies of components used in gateway-105.
They are reference building blocks, not a complete deployment launcher.

- gateway.py: accepts deployment-supplied worker-identity/readiness callbacks; does not retry HTTP business failures.
- tls_context.py: explicit CA trust with hostname and certificate verification.
- watchdog.py: restart policy and supervision logic for an owned controller. Its executable entry expects the original project-specific operations.py and private state layout, which are intentionally not distributed. Use its policy functions with deployment-specific integration; do not launch it as a standalone tool.

The original site credentials, worker coordination state, endpoint routing, and benchmark database are not included. See [architecture](../docs/ARCHITECTURE.md).

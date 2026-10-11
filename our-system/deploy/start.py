"""Single-worker start; disable request access logs and untrusted forwarding."""
import os
import uvicorn


if __name__ == "__main__":
    uvicorn.run("deploy.runtime:create_app", factory=True, host=os.environ.get("AML_HOST", "127.0.0.1"),
                port=int(os.environ.get("AML_PORT", "8000")), workers=1,
                access_log=False, proxy_headers=False, timeout_keep_alive=5,
                timeout_graceful_shutdown=45)

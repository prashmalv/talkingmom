"""Launcher.

    python run.py           → http://localhost:8000
    python run.py --https   → https://<this-PC's-IP>:8443 (self-signed certificate)

HTTPS is needed when other phones/laptops join a Phase 2 meeting room:
browsers only allow the microphone on https:// pages (or on localhost).
"""
import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import uvicorn

# On Windows, redirected stdout (log file, IDE, launcher) defaults to cp1252,
# which can't print the ✓/✗ preflight marks or Devanagari and crashes startup.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CERT_DIR = Path(__file__).resolve().parent / "certs"


class _RealAddresses(logging.Filter):
    """Uvicorn announces 'running on http(s)://0.0.0.0:PORT' — 0.0.0.0 only means
    'every network interface' and can't be opened in a browser. Print the
    addresses people should actually open instead."""

    def __init__(self, scheme: str, port: int, ip: str):
        super().__init__()
        self.text = (f"Open {scheme}://localhost:{port}/ on this PC · "
                     f"{scheme}://{ip}:{port}/ from other devices on the same network "
                     f"(Phase 2: /phase2, participants: /join) — press CTRL+C to quit")

    def filter(self, record: logging.LogRecord) -> bool:
        if "Uvicorn running on" in str(record.msg):
            record.msg, record.args = self.text, ()
            record.color_message = self.text  # uvicorn's coloured formatter prints this copy
        return True


class _HideConnectionReset(logging.Filter):
    """Windows asyncio logs a traceback whenever a browser drops a connection
    (e.g. at the self-signed certificate warning). It's harmless noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, ConnectionResetError)


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packet is sent; just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), r"C:\Program Files (x86)"):
        p = Path(base) / "Git" / "usr" / "bin" / "openssl.exe"
        if p.exists():
            return str(p)
    return None


def _self_signed_cert(ip: str) -> tuple[str, str]:
    """Create (once) a self-signed certificate for this PC's LAN address."""
    key, crt = CERT_DIR / "dev.key", CERT_DIR / "dev.crt"
    if key.exists() and crt.exists():
        return str(key), str(crt)
    openssl = _find_openssl()
    if not openssl:
        sys.exit("openssl not found — install Git for Windows (it includes openssl) or put openssl on PATH.")
    CERT_DIR.mkdir(exist_ok=True)
    subprocess.run([
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
        "-keyout", str(key), "-out", str(crt), "-subj", "/CN=MoM local",
        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
    ], check=True, capture_output=True)
    return str(key), str(crt)


if __name__ == "__main__":
    ip = _lan_ip()
    logging.getLogger("asyncio").addFilter(_HideConnectionReset())
    if "--https" in sys.argv:
        key, crt = _self_signed_cert(ip)
        print("=" * 72, flush=True)
        print(f"HTTPS — this PC:       https://localhost:8443/phase2", flush=True)
        print(f"        other devices: https://{ip}:8443/join", flush=True)
        print("Browsers warn about the self-signed certificate once:", flush=True)
        print("choose 'Advanced' → 'Proceed'. (Don't open 0.0.0.0 — it isn't an address.)", flush=True)
        print("=" * 72, flush=True)
        logging.getLogger("uvicorn.error").addFilter(_RealAddresses("https", 8443, ip))
        uvicorn.run("app.main:app", host="0.0.0.0", port=8443, reload=False,
                    ssl_keyfile=key, ssl_certfile=crt)
    else:
        logging.getLogger("uvicorn.error").addFilter(_RealAddresses("http", 8000, ip))
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

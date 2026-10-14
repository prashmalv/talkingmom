"""Launcher: `python run.py` — starts uvicorn on http://localhost:8000"""
import sys

import uvicorn

# On Windows, redirected stdout (log file, IDE, launcher) defaults to cp1252,
# which can't print the ✓/✗ preflight marks or Devanagari and crashes startup.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

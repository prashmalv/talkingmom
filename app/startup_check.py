"""Startup preflight checks — run once when the server boots.

Prints a clear ✓/✗ report for every dependency, credential, file, and API,
with actionable fix suggestions for any failure.

Non-blocking: failures print but do not crash the server. This lets you at
least see what's broken and fix it, rather than hitting cryptic errors later.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_POLISH,
    AZURE_OPENAI_DEPLOYMENT_SUMMARY,
    AZURE_OPENAI_DEPLOYMENT_TRANSLATE,
    AZURE_OPENAI_ENDPOINT,
    BASE_DIR,
    CHROMA_DIR,
    ENROLL_DIR,
    HF_TOKEN,
    MEETINGS_DIR,
    OPENAI_API_KEY,
    UPLOAD_DIR,
    USE_AZURE_OPENAI,
    WHISPER_MODEL,
)

# ---- pretty print helpers -------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_OK = f"{_GREEN}✓{_RESET}"
_FAIL = f"{_RED}✗{_RESET}"
_WARN = f"{_YELLOW}⚠{_RESET}"


class Result:
    __slots__ = ("name", "ok", "detail", "fix")
    def __init__(self, name: str, ok: bool | None, detail: str = "", fix: str = ""):
        self.name = name
        self.ok = ok           # True=pass, False=fail, None=warn
        self.detail = detail
        self.fix = fix

    def render(self) -> str:
        mark = _OK if self.ok is True else (_FAIL if self.ok is False else _WARN)
        line = f"  {mark} {self.name}"
        if self.detail:
            line += f"  {_BLUE}{self.detail}{_RESET}"
        return line


# ---- individual checks ----------------------------------------------------

def _check_python() -> Result:
    v = sys.version_info
    ok = v >= (3, 10)
    return Result(
        "Python version",
        ok,
        detail=f"{v.major}.{v.minor}.{v.micro}",
        fix="Upgrade to Python 3.10 or newer." if not ok else "",
    )


def _check_ffmpeg() -> Result:
    exe = shutil.which("ffmpeg")
    if not exe:
        return Result(
            "ffmpeg installed",
            False,
            "not found on PATH",
            "Install ffmpeg and ensure it's on PATH. Windows: `winget install ffmpeg`. "
            "Without ffmpeg, MP3/M4A/WebM audio can't be loaded.",
        )
    try:
        out = subprocess.check_output([exe, "-version"], text=True, timeout=5)
        ver = out.splitlines()[0].strip()
        return Result("ffmpeg installed", True, ver)
    except Exception as e:
        return Result("ffmpeg installed", None, f"found but not runnable ({e})",
                      "Check `ffmpeg -version` runs manually in your shell.")


def _check_dirs() -> Result:
    dirs = {"UPLOAD_DIR": UPLOAD_DIR, "ENROLL_DIR": ENROLL_DIR,
            "MEETINGS_DIR": MEETINGS_DIR, "CHROMA_DIR": CHROMA_DIR}
    problems = []
    for name, p in dirs.items():
        try:
            p.mkdir(parents=True, exist_ok=True)
            # writable?
            test_file = p / ".startup_check.tmp"
            test_file.write_text("ok")
            test_file.unlink()
        except Exception as e:
            problems.append(f"{name}={p} ({e})")
    if problems:
        return Result(
            "Data directories writable",
            False,
            "; ".join(problems),
            "Check filesystem permissions on the data folders, or override with env vars.",
        )
    return Result("Data directories writable", True,
                  f"{len(dirs)} folders under {BASE_DIR / 'data'}")


def _check_hf_token() -> Result:
    if not HF_TOKEN:
        return Result(
            "HuggingFace token",
            False,
            "HF_TOKEN not set in .env",
            "Get a token at https://huggingface.co/settings/tokens (Read scope) "
            "and add `HF_TOKEN=hf_xxx` to your .env",
        )
    if not HF_TOKEN.startswith("hf_"):
        return Result(
            "HuggingFace token", None,
            f"present but doesn't look like a HF token (starts with '{HF_TOKEN[:4]}')",
            "HF tokens start with 'hf_'. Regenerate at https://huggingface.co/settings/tokens",
        )
    return Result("HuggingFace token", True, f"hf_...{HF_TOKEN[-4:]}")


def _check_openai_config() -> Result:
    if not OPENAI_API_KEY:
        return Result(
            "LLM credentials",
            False,
            "OPENAI_API_KEY not set",
            "Add OPENAI_API_KEY to .env (regular OpenAI 'sk-...' key OR your Azure API key).",
        )
    if USE_AZURE_OPENAI:
        # Azure — need deployment names too.
        missing = []
        if not AZURE_OPENAI_DEPLOYMENT_TRANSLATE and not AZURE_OPENAI_DEPLOYMENT_SUMMARY \
                and not AZURE_OPENAI_DEPLOYMENT_POLISH:
            missing.append("no AZURE_OPENAI_DEPLOYMENT_* set")
        if missing:
            return Result(
                "LLM credentials",
                False,
                f"Azure endpoint set but {missing[0]}",
                "Add at least AZURE_OPENAI_DEPLOYMENT_SUMMARY=<your-deployment-name> "
                "(from Azure Portal → your resource → Model deployments).",
            )
        deps = ", ".join(sorted({AZURE_OPENAI_DEPLOYMENT_TRANSLATE,
                                 AZURE_OPENAI_DEPLOYMENT_POLISH,
                                 AZURE_OPENAI_DEPLOYMENT_SUMMARY} - {""}))
        return Result("LLM credentials", True,
                      f"Azure OpenAI @ {AZURE_OPENAI_ENDPOINT} (deployments: {deps})")
    return Result("LLM credentials", True, "OpenAI (platform.openai.com)")


def _check_openai_live() -> Result:
    """Actually hit the API with a tiny ping to confirm auth works."""
    if not OPENAI_API_KEY:
        return Result("LLM API reachable", None, "skipped (no key)")
    try:
        from .llm_client import llm_chat
        t0 = time.time()
        out = llm_chat(
            "summary",
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
            temperature=0,
        )
        ms = int((time.time() - t0) * 1000)
        if "pong" in out.lower():
            return Result("LLM API reachable", True,
                          f"replied in {ms}ms via {'Azure' if USE_AZURE_OPENAI else 'OpenAI'}")
        return Result("LLM API reachable", None,
                      f"got a reply but not 'pong' ({out[:60]!r})",
                      "API works but response was unexpected — model behavior only.")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg or "invalid" in msg.lower():
            fix = ("Wrong API key for this endpoint. Regenerate the key in your "
                   "OpenAI/Azure resource, or check that OPENAI_API_KEY matches "
                   "AZURE_OPENAI_ENDPOINT's resource.")
        elif "404" in msg or "DeploymentNotFound" in msg or "not found" in msg.lower():
            fix = ("Deployment name doesn't exist. Check AZURE_OPENAI_DEPLOYMENT_* "
                   "matches the 'Deployment name' column in Azure → Model deployments.")
        elif "PermissionDenied" in msg or "403" in msg:
            fix = "Key valid but lacks permission for this deployment. Ask your Azure admin."
        else:
            fix = "See full error above; check endpoint + key + deployment names."
        return Result("LLM API reachable", False, msg.splitlines()[0][:150], fix)


def _check_whisper() -> Result:
    """Verify faster-whisper is installed and model config is sane. Does NOT load
    the model here (that would slow startup by minutes on first run)."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return Result(
            "faster-whisper installed", False, "package missing",
            "Run: pip install faster-whisper==1.0.3",
        )
    return Result("Whisper transcription", True,
                  f"model={WHISPER_MODEL} (loads on first meeting)")


def _check_pyannote() -> Result:
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return Result(
            "pyannote-audio installed", False, "package missing",
            "Run: pip install pyannote.audio==3.3.2",
        )
    if not HF_TOKEN:
        return Result(
            "pyannote diarization", False,
            "HF_TOKEN missing (required to download diarization model)",
            "See HuggingFace token check above.",
        )
    return Result("pyannote diarization", True, "3.1 (loads on first meeting)")


def _check_speechbrain() -> Result:
    try:
        import speechbrain  # noqa: F401
        return Result("speechbrain (voice embeddings)", True, "ECAPA-TDNN ready")
    except ImportError:
        return Result(
            "speechbrain installed", False, "package missing",
            "Run: pip install speechbrain==1.0.2",
        )


def _check_chromadb() -> Result:
    try:
        from .speaker_db import list_speakers
        n = len(list_speakers())
        return Result("Speaker DB (ChromaDB)", True, f"{n} enrolled voices")
    except Exception as e:
        return Result(
            "Speaker DB (ChromaDB)", False, str(e).splitlines()[0][:120],
            "Check CHROMA_DIR is writable and chromadb package installed.",
        )


def _check_frontend() -> Result:
    fp = BASE_DIR / "frontend" / "index.html"
    if not fp.exists():
        return Result(
            "Frontend files", False, f"missing {fp}",
            "Restore frontend/index.html — see git history or backup.",
        )
    size_kb = fp.stat().st_size // 1024
    return Result("Frontend files", True, f"index.html ({size_kb} KB)")


# ---- runner ---------------------------------------------------------------

def run_startup_checks() -> None:
    """Run all checks and print a formatted report to stdout."""
    line = "=" * 72
    print(line, flush=True)
    print(f"{_BOLD}Startup preflight checks{_RESET}", flush=True)
    print(line, flush=True)

    checks = [
        _check_python,
        _check_ffmpeg,
        _check_dirs,
        _check_frontend,
        _check_hf_token,
        _check_pyannote,
        _check_speechbrain,
        _check_whisper,
        _check_chromadb,
        _check_openai_config,
        _check_openai_live,
    ]

    results: list[Result] = []
    for check in checks:
        try:
            r = check()
        except Exception as e:
            r = Result(check.__name__.replace("_check_", ""), False,
                       f"unexpected error: {e}",
                       "Check server logs for the full traceback.")
            traceback.print_exc()
        results.append(r)
        print(r.render(), flush=True)

    # Print fix suggestions for any failures/warnings.
    problems = [r for r in results if r.ok is not True]
    if problems:
        print("", flush=True)
        print(f"{_BOLD}{_YELLOW}Action items:{_RESET}", flush=True)
        for r in problems:
            if r.fix:
                mark = _FAIL if r.ok is False else _WARN
                print(f"  {mark} {_BOLD}{r.name}{_RESET}: {r.fix}", flush=True)
        print("", flush=True)

    fails = sum(1 for r in results if r.ok is False)
    warns = sum(1 for r in results if r.ok is None)
    passes = sum(1 for r in results if r.ok is True)

    summary = f"{passes} passed"
    if warns:
        summary += f", {_YELLOW}{warns} warning{'s' if warns > 1 else ''}{_RESET}"
    if fails:
        summary += f", {_RED}{fails} failed{_RESET}"

    print(line, flush=True)
    print(f"Result: {summary}", flush=True)
    if fails == 0 and warns == 0:
        print(f"{_GREEN}All systems ready — process a meeting to test end-to-end.{_RESET}", flush=True)
    elif fails == 0:
        print(f"{_YELLOW}Server can start, but some non-critical checks warn — see above.{_RESET}", flush=True)
    else:
        print(f"{_RED}Server started but some features WILL fail until the errors above are fixed.{_RESET}", flush=True)
    print(line, flush=True)


# Enable ANSI colors on Windows terminals that support them.
if os.name == "nt":
    try:
        import colorama  # type: ignore
        colorama.just_fix_windows_console()
    except ImportError:
        # Try enabling VT mode via ctypes as a fallback.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            # Strip colors if we can't enable ANSI support.
            _GREEN = _RED = _YELLOW = _BLUE = _RESET = _BOLD = ""
            _OK, _FAIL, _WARN = "[OK]", "[FAIL]", "[WARN]"

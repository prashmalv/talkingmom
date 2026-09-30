import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# Accept both names — Azure users often set AZURE_OPENAI_API_KEY.
OPENAI_API_KEY = (
    os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Azure OpenAI support (auto-detected when AZURE_OPENAI_ENDPOINT is set).
_raw_azure = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
# Some users paste the "OpenAI-compatible v1 endpoint" (…/openai/v1) from Azure
# AI Foundry. The AzureOpenAI SDK adds `/openai/deployments/...` itself, so
# strip any trailing /openai or /openai/v1 to avoid double paths.
for _suffix in ("/openai/v1", "/openai"):
    if _raw_azure.lower().endswith(_suffix):
        _raw_azure = _raw_azure[: -len(_suffix)].rstrip("/")
        break
AZURE_OPENAI_ENDPOINT = _raw_azure
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
def _clean_env(name: str) -> str:
    """Strip whitespace and surrounding quotes from an env var."""
    v = os.getenv(name, "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1].strip()
    return v


AZURE_OPENAI_DEPLOYMENT_TRANSLATE = _clean_env("AZURE_OPENAI_DEPLOYMENT_TRANSLATE")
AZURE_OPENAI_DEPLOYMENT_POLISH = _clean_env("AZURE_OPENAI_DEPLOYMENT_POLISH")
AZURE_OPENAI_DEPLOYMENT_SUMMARY = _clean_env("AZURE_OPENAI_DEPLOYMENT_SUMMARY")

# True when running against Azure (vs. platform.openai.com).
USE_AZURE_OPENAI = bool(AZURE_OPENAI_ENDPOINT)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")           # cpu | cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 (fast, CPU), int8_float16, float16 (GPU)
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
# Optional per-language model overrides — set to a HF model id to use a fine-tuned model.
# e.g. WHISPER_MODEL_MR=/whisvasista22per-marathi-medium
WHISPER_MODEL_MR = os.getenv("WHISPER_MODEL_MR", "").strip()
WHISPER_MODEL_HI = os.getenv("WHISPER_MODEL_HI", "").strip()
POLISH_TRANSCRIPT = os.getenv("POLISH_TRANSCRIPT", "false").strip().lower() in ("1", "true", "yes")

# Separate models for polish (cheap) vs translation (quality).
GPT_POLISH_MODEL = os.getenv("GPT_POLISH_MODEL", "gpt-4o-mini")
GPT_TRANSLATE_MODEL = os.getenv("GPT_TRANSLATE_MODEL", "gpt-4o")
SPEAKER_MATCH_THRESHOLD = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.35"))

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
ENROLL_DIR = Path(os.getenv("ENROLL_DIR", BASE_DIR / "data" / "enrollments"))
MEETINGS_DIR = Path(os.getenv("MEETINGS_DIR", BASE_DIR / "data" / "meetings"))

for d in (CHROMA_DIR, UPLOAD_DIR, ENROLL_DIR, MEETINGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPEAKER_COLLECTION = "speaker_profiles"

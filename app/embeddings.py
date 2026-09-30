"""SpeechBrain ECAPA-TDNN speaker embeddings.

SpeechBrain 1.0.2 hardcodes symlink creation in a few spots, which fails on
Windows without Developer Mode (WinError 1314). We monkey-patch its linking
helper to use plain copies before importing anything that triggers it.
"""
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# --- monkey-patch speechbrain's symlink helper BEFORE the classifier is loaded ---
import speechbrain.utils.fetching as _sb_fetch


def _copy_instead_of_symlink(src, dst, local_strategy=None):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except (shutil.SameFileError, OSError):
        pass
    return dst


_sb_fetch.link_with_strategy = _copy_instead_of_symlink

from huggingface_hub import snapshot_download  # noqa: E402
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

from .config import BASE_DIR  # noqa: E402

_HF_REPO = "speechbrain/spkrec-ecapa-voxceleb"
_MODEL_DIR = BASE_DIR / "data" / "models" / "ecapa"


def _ensure_model() -> Path:
    return Path(snapshot_download(repo_id=_HF_REPO))


@lru_cache(maxsize=1)
def get_encoder() -> EncoderClassifier:
    model_dir = _ensure_model()
    return EncoderClassifier.from_hparams(
        source=str(model_dir),
        savedir=str(model_dir),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )


def embed(wav: np.ndarray) -> np.ndarray:
    """Return a unit-normalized 192-d embedding for the given waveform."""
    encoder = get_encoder()
    if wav.size < 1600:  # ~0.1s minimum
        return np.zeros(192, dtype=np.float32)
    tensor = torch.from_numpy(wav).float().unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(tensor).squeeze().cpu().numpy()
    emb = emb.astype(np.float32)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb

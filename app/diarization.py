"""pyannote.audio speaker diarization."""
from functools import lru_cache
from pathlib import Path
from pyannote.audio import Pipeline
import torch

from .config import HF_TOKEN


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Get one at https://huggingface.co/settings/tokens "
            "and accept the pyannote/speaker-diarization-3.1 model terms."
        )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def diarize(audio_path: str | Path) -> list[dict]:
    """Returns list of {start, end, speaker} sorted by start time."""
    pipeline = get_pipeline()
    diarization = pipeline(str(audio_path))
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        )
    turns.sort(key=lambda t: t["start"])
    return turns

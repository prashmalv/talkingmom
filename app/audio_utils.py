"""Audio loading and slicing helpers. Everything normalizes to 16 kHz mono."""
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    wav, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32), sr


def slice_audio(wav: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(start_s * sr))
    b = min(len(wav), int(end_s * sr))
    return wav[a:b]


def save_wav(path: str | Path, wav: np.ndarray, sr: int = TARGET_SR) -> None:
    sf.write(str(path), wav, sr, subtype="PCM_16")

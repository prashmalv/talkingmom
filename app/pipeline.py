"""End-to-end pipeline: diarize -> per-segment (embed + STT + speaker match) -> assemble."""
from pathlib import Path
import numpy as np

from .audio_utils import load_audio, slice_audio
from .diarization import diarize
from .transcription import gpt_polish_batch, gpt_translate_batch, transcribe_segment, translate_segment
from .config import OPENAI_API_KEY, POLISH_TRANSCRIPT
from .embeddings import embed
from .speaker_db import get_speaker_language_by_name, match_speaker


def _merge_consecutive(segments: list[dict]) -> list[dict]:
    """Merge back-to-back segments from the same resolved speaker."""
    merged: list[dict] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
            if "translation" in seg:
                merged[-1]["translation"] = (
                    merged[-1].get("translation", "") + " " + seg.get("translation", "")
                ).strip()
        else:
            merged.append(dict(seg))
    return merged


def process_meeting(
    audio_path: str | Path,
    language: str | None = None,
    translate: bool = False,
    progress_cb=None,
) -> dict:
    def _p(pct: int, phase: str | None = None):
        if progress_cb:
            try:
                progress_cb(pct, phase)
            except Exception:
                pass
    _p(5, "loading")
    wav, sr = load_audio(audio_path)

    _p(10, "diarization")
    turns = diarize(audio_path)
    if not turns:
        _p(100, "done")
        return {"segments": [], "transcript": "", "unresolved": []}
    _p(25, "diarization")

    # One embedding per anonymous pyannote label, aggregated over all its turns.
    label_wavs: dict[str, list[np.ndarray]] = {}
    for t in turns:
        label_wavs.setdefault(t["speaker"], []).append(
            slice_audio(wav, sr, t["start"], t["end"])
        )

    label_to_name: dict[str, str] = {}
    label_to_lang: dict[str, str | None] = {}
    unresolved: list[dict] = []
    for label, chunks in label_wavs.items():
        merged = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        # Cap at 30s of audio to keep the embedding call bounded.
        if len(merged) > 30 * sr:
            merged = merged[: 30 * sr]
        emb = embed(merged)
        name, dist = match_speaker(emb)
        resolved = name or f"Unknown ({label})"
        label_to_name[label] = resolved
        # Per-speaker language: use enrolled language when the speaker is recognized,
        # otherwise fall back to the user-selected language (or auto).
        label_to_lang[label] = get_speaker_language_by_name(name) if name else None
        if name is None:
            unresolved.append({"label": label, "closest_distance": dist})

    _p(30, "transcription")
    segments = []
    total = len(turns)
    for i, t in enumerate(turns):
        chunk = slice_audio(wav, sr, t["start"], t["end"])
        # Per-speaker language beats the fallback selected in the UI.
        seg_lang = label_to_lang.get(t["speaker"]) or language
        text = transcribe_segment(chunk, language=seg_lang)
        if not text:
            continue
        seg = {
            "start": round(t["start"], 2),
            "end": round(t["end"], 2),
            "speaker": label_to_name[t["speaker"]],
            "raw_speaker": t["speaker"],
            "text": text,
            "language": seg_lang or "auto",
        }
        segments.append(seg)
        # 30% -> 88% during transcription
        _p(30 + int(58 * (i + 1) / total), "transcription")

    _p(90, "finalizing")
    segments = _merge_consecutive(segments)

    # Group segments by language once for both polish + translate.
    by_lang_pre: dict[str, list[int]] = {}
    for idx, s in enumerate(segments):
        lang_i = (s.get("language") or "").lower()
        if lang_i and lang_i != "en":
            by_lang_pre.setdefault(lang_i, []).append(idx)

    # Polish ASR errors in non-English transcripts before translation.
    if POLISH_TRANSCRIPT and OPENAI_API_KEY and by_lang_pre:
        _p(91, "polishing")
        for lang, indices in by_lang_pre.items():
            texts = [segments[i]["text"] for i in indices]
            try:
                fixed = gpt_polish_batch(texts, lang)
                print(f"[Pipeline] Polished {len(indices)} {lang} segments via GPT-4o-mini", flush=True)
                for i, t in zip(indices, fixed):
                    if t:
                        segments[i]["text"] = t
            except Exception as e:
                print(f"[Pipeline] Polish failed ({e}); using raw ASR text", flush=True)

    # Batch-translate AFTER merging so GPT sees full context. Group by language,
    # one API call per language, numbered lines round-trip preserved.
    if translate:
        _p(93, "translation")
        by_lang: dict[str, list[int]] = {}
        for idx, s in enumerate(segments):
            lang = (s.get("language") or "").lower()
            if lang and lang != "en":
                by_lang.setdefault(lang, []).append(idx)

        for lang, indices in by_lang.items():
            texts = [segments[i]["text"] for i in indices]
            translations: list[str] = []
            if OPENAI_API_KEY:
                try:
                    translations = gpt_translate_batch(texts, lang)
                    print(f"[Pipeline] Batch-translated {len(indices)} {lang} segments via GPT", flush=True)
                except Exception as e:
                    print(f"[Pipeline] Batch translation failed ({e}); leaving translation blank", flush=True)
                    translations = []
            for i, tr in zip(indices, translations):
                if tr:
                    segments[i]["translation"] = tr
    transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
    return {
        "segments": segments,
        "transcript": transcript,
        "unresolved": unresolved,
    }

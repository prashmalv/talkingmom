"""End-to-end pipeline.

    load audio
      ├─ thread: Whisper over the whole file (word timestamps)
      └─ pyannote diarization → non-overlapping turns → identify each turn's voice
    words → turns by timestamp → merge same-speaker turns → polish → translate
"""
from bisect import bisect_right
from pathlib import Path
import threading
import time

import numpy as np

from .audio_utils import load_audio, slice_audio
from .diarization import diarize
from .transcription import (
    detect_language_prob,
    gpt_polish_batch,
    gpt_translate_batch,
    transcribe_segment,
    transcribe_words,
)
from .config import (
    DIARIZATION_STEP,
    OPENAI_API_KEY,
    POLISH_TRANSCRIPT,
    SPEAKER_CLUSTER_SIMILARITY,
    SPEAKER_MATCH_LOOSE,
    SPEAKER_MATCH_MARGIN,
    SPEAKER_MATCH_THRESHOLD,
)
from .embeddings import embed
from .llm_client import model_label
from .run_log import MeetingLog
from .speaker_db import get_speaker_language_by_name, list_names, normalize_lang, rank_names


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


_MIN_TURN_S = 0.25


def _exclusive_turns(turns: list[dict]) -> list[dict]:
    """Remove overlaps between diarization turns.

    pyannote emits overlapping turns when people talk over each other. Slicing
    each turn independently makes Whisper transcribe the shared audio twice,
    so the same sentence appears under two speakers. Here every instant is
    given to exactly one speaker: in an overlap, the speaker who started most
    recently wins. Slivers shorter than _MIN_TURN_S are dropped.
    """
    bounds = sorted({t["start"] for t in turns} | {t["end"] for t in turns})
    out: list[dict] = []
    for a, b in zip(bounds, bounds[1:]):
        active = [t for t in turns if t["start"] <= a and t["end"] >= b]
        if not active:
            continue
        owner = max(active, key=lambda t: t["start"])["speaker"]
        if out and out[-1]["speaker"] == owner and abs(out[-1]["end"] - a) < 1e-6:
            out[-1]["end"] = b
        else:
            out.append({"start": a, "end": b, "speaker": owner})
    return [t for t in out if t["end"] - t["start"] >= _MIN_TURN_S]


# A turn this long carries enough voice to check its language on its own.
_TURN_ID_MIN_S = 1.5
# Turns at least this long have a voiceprint reliable enough to drive grouping;
# shorter ones ("Yes.", "Perfect.") are attached to a group afterwards.
_CLUSTER_MIN_S = 1.0
# Seconds of processing per second of audio, refined after each meeting;
# drives the progress bar while transcription runs.
_rtf = {"value": 1.2}


def _group_by_voice(turns: list[dict], wav: np.ndarray, sr: int, log=print) -> None:
    """Re-group turns into speakers by their ECAPA voiceprints (rewrites t["speaker"]).

    pyannote's own grouping is unreliable on short, alternating conversations:
    it can put two people under one label (or one person under two). Its turn
    *boundaries* are good, so we keep those and redo the grouping: average-
    linkage clustering of per-turn voiceprints, stopping when groups are less
    than SPEAKER_CLUSTER_SIMILARITY alike.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    embs = {i: embed(slice_audio(wav, sr, t["start"], t["end"]))
            for i, t in enumerate(turns) if t["end"] - t["start"] >= 0.3}
    long_ids = [i for i in embs if turns[i]["end"] - turns[i]["start"] >= _CLUSTER_MIN_S]
    if len(long_ids) < 2:
        log("  too little speech to re-group by voice; keeping pyannote's groups")
        return

    E = np.stack([embs[i] for i in long_ids])
    ids = fcluster(linkage(E, method="average", metric="cosine"),
                   t=1 - SPEAKER_CLUSTER_SIMILARITY, criterion="distance")
    assign = {i: int(g) for i, g in zip(long_ids, ids)}
    centroids = {}
    for g in set(assign.values()):
        c = np.mean([embs[i] for i, gg in assign.items() if gg == g], axis=0)
        centroids[g] = c / (np.linalg.norm(c) or 1.0)

    # pyannote label -> group most of its long turns landed in (fallback vote).
    votes: dict[str, dict[int, int]] = {}
    for i, g in assign.items():
        v = votes.setdefault(turns[i]["speaker"], {})
        v[g] = v.get(g, 0) + 1

    for i, t in enumerate(turns):
        if i in assign:
            continue
        g = None
        if i in embs:
            sims = {gg: float(embs[i] @ c) for gg, c in centroids.items()}
            best = max(sims, key=sims.get)
            if sims[best] >= 0.25:
                g = best
        if g is None and t["speaker"] in votes:
            g = max(votes[t["speaker"]], key=votes[t["speaker"]].get)
        if g is None:  # nearest grouped turn in time
            g = assign[min(long_ids, key=lambda j: abs(turns[j]["start"] - t["start"]))]
        assign[i] = g

    before = len({t["speaker"] for t in turns})
    new_label: dict[int, str] = {}
    for i, t in enumerate(turns):
        t["speaker"] = new_label.setdefault(assign[i], f"SPEAKER_{len(new_label):02d}")
    log(f"  voiceprints grouped {len(turns)} turns into {len(new_label)} voices (pyannote had {before})")


def _identify(turns: list[dict], wav: np.ndarray, sr: int, log=print) -> list[dict]:
    """Group turns by voice, then name each group (sets t["name"]).

    Each group's pooled audio (up to 60 s) is compared with every enrolled
    person — far more reliable than judging single turns. A group gets a name
    if it's within SPEAKER_MATCH_THRESHOLD, or within SPEAKER_MATCH_LOOSE and
    clearly ahead (SPEAKER_MATCH_MARGIN) of the next-closest person. Each
    person can be given to only one group per meeting.
    Returns the groups that stayed unknown.
    """
    _group_by_voice(turns, wav, sr, log)

    by_label: dict[str, list[np.ndarray]] = {}
    for t in turns:
        by_label.setdefault(t["speaker"], []).append(slice_audio(wav, sr, t["start"], t["end"]))
    ranks_of = {label: rank_names(embed(np.concatenate(chunks)[: 60 * sr]))
                for label, chunks in by_label.items()}

    # Best matches first, one person per group and one group per person.
    candidates = []
    for label, ranks in ranks_of.items():
        for name, dist in ranks:
            others = [d for n, d in ranks if n != name]
            margin = (min(others) if others else 1.0) - dist
            if dist <= SPEAKER_MATCH_THRESHOLD or (dist <= SPEAKER_MATCH_LOOSE and margin >= SPEAKER_MATCH_MARGIN):
                candidates.append((dist, label, name, margin))
    label_name: dict[str, tuple[str, float, float]] = {}
    used: set[str] = set()
    for dist, label, name, margin in sorted(candidates):
        if label not in label_name and name not in used:
            label_name[label] = (name, dist, margin)
            used.add(name)

    unresolved: list[dict] = []
    for label, chunks in by_label.items():
        secs = sum(len(c) for c in chunks) / sr
        ranks = ranks_of[label]
        if label in label_name:
            name, dist, margin = label_name[label]
            how = "" if dist <= SPEAKER_MATCH_THRESHOLD else f", weaker match but {margin:.2f} ahead of next person"
            log(f"  {label} ({secs:.0f}s of speech) → {name} (distance {dist:.2f}{how})")
        else:
            closest = f"closest {ranks[0][0]} {ranks[0][1]:.2f}" if ranks else "no one enrolled"
            unresolved.append({"label": label, "closest_distance": ranks[0][1] if ranks else None})
            log(f"  {label} ({secs:.0f}s of speech) → Unknown ({closest})")

    for t in turns:
        t["name"] = label_name[t["speaker"]][0] if t["speaker"] in label_name else f"Unknown ({t['speaker']})"
    return unresolved


_SENT_END = (".", "?", "!", "।")


def _fix_boundaries(per_turn: list[list[dict]], turns: list[dict]) -> None:
    """Move a sentence tail that spilled into the next speaker's turn back.

    Diarization boundaries can land a few hundred ms early, so the last word
    of one person's sentence ("…go on a | picnic? That's a great idea") lands
    in the next turn. If a turn ends mid-sentence and the next speaker's turn
    opens with ≤3 words finishing that sentence right at the boundary, those
    words move back.
    """
    for i in range(len(turns) - 1):
        cur, nxt = per_turn[i], per_turn[i + 1]
        if not cur or not nxt or turns[i]["name"] == turns[i + 1]["name"]:
            continue
        if cur[-1]["word"].strip().endswith(_SENT_END):
            continue
        for k, w in enumerate(nxt[:3]):
            if w["start"] - turns[i]["end"] > 1.0:
                break
            if w["word"].strip().endswith(_SENT_END):
                cur.extend(nxt[: k + 1])
                del nxt[: k + 1]
                # Keep per-line playback in sync with the moved words.
                turns[i]["end"] = max(turns[i]["end"], w["end"])
                turns[i + 1]["start"] = min(max(turns[i + 1]["start"], w["end"]), turns[i + 1]["end"])
                break


def _assign_words(words: list[dict], turns: list[dict]) -> list[str]:
    """Give each word to the turn containing its midpoint (else the nearest turn)."""
    out: list[list[dict]] = [[] for _ in turns]
    starts = [t["start"] for t in turns]

    def gap(j: int, x: float) -> float:
        t = turns[j]
        return 0.0 if t["start"] <= x <= t["end"] else min(abs(x - t["start"]), abs(x - t["end"]))

    for w in words:
        mid = (w["start"] + w["end"]) / 2
        i = bisect_right(starts, mid) - 1
        cands = [j for j in (i, i + 1) if 0 <= j < len(turns)]
        out[min(cands, key=lambda j: gap(j, mid))].append(w)
    _fix_boundaries(out, turns)
    return ["".join(w["word"] for w in ws).strip() for ws in out]


def process_meeting(
    audio_path: str | Path,
    language: str | None = None,
    translate: bool = False,
    progress_cb=None,
    log: MeetingLog | None = None,
) -> dict:
    def _p(pct: int, phase: str | None = None):
        if progress_cb:
            try:
                progress_cb(pct, phase)
            except Exception:
                pass

    log = log or MeetingLog()
    _p(5, "loading")
    wav, sr = load_audio(audio_path)
    duration = len(wav) / sr
    names = list_names()
    t0 = time.time()
    log(f"Audio: {Path(audio_path).name}, {duration:.1f}s, language={language or 'auto'}, "
        f"translate={'on' if translate else 'off'}, polish={'on' if POLISH_TRANSCRIPT and OPENAI_API_KEY else 'off'}")
    log(f"Enrolled voices: {', '.join(names) if names else '(none)'}")

    # Whisper and pyannote don't depend on each other — run them concurrently.
    stt: dict = {}

    def _stt():
        t = time.time()
        try:
            stt["res"] = transcribe_words(wav, language, names, log=log)
            if stt["res"]:
                log(f"✓ Whisper done in {time.time() - t:.1f}s — {len(stt['res']['words'])} words "
                    f"('{stt['res']['model']}', {stt['res']['language']})")
        except Exception as e:  # reported below; per-turn fallback takes over
            stt["err"] = e

    log("Step 1 — Speech-to-text (Whisper) started in background")
    stt_thread = threading.Thread(target=_stt, daemon=True)
    stt_thread.start()

    # Smooth progress (10% -> 85%) while the heavy models run, estimated from
    # how fast the previous meeting went.
    done = threading.Event()

    def _ticker():
        est = max(8.0, duration * _rtf["value"])
        while not done.wait(0.5):
            _p(10 + int(75 * min(0.97, (time.time() - t0) / est)), "transcription")

    threading.Thread(target=_ticker, daemon=True).start()

    try:
        with log.step(f"Step 2 — Who spoke when (pyannote/speaker-diarization-3.1, "
                      f"window step {10 * DIARIZATION_STEP:.1f}s)"):
            raw = diarize(audio_path)
            turns = _exclusive_turns(raw)
            labels = sorted({t["speaker"] for t in turns})
            log(f"  {len(turns)} speaking turns, {len(labels)} distinct voices: {', '.join(labels)}")
        if turns:
            with log.step(f"Step 3 — Group turns by voice & identify (SpeechBrain ECAPA vs ChromaDB, "
                          f"match ≤{SPEAKER_MATCH_THRESHOLD}, or ≤{SPEAKER_MATCH_LOOSE} if clearly ahead)"):
                unresolved = _identify(turns, wav, sr, log=log)
        else:
            unresolved = []
        if stt_thread.is_alive():
            log("Waiting for Whisper to finish …")
        stt_thread.join()
    finally:
        done.set()
    if duration > 5:
        _rtf["value"] = max(0.3, (time.time() - t0) / duration)

    if not turns:
        log("No speech found — nothing to transcribe")
        _p(100, "done")
        return {"segments": [], "transcript": "", "unresolved": []}

    segments = build_transcript(wav, sr, turns, stt, language, translate, names, log, _p, t0)
    transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
    return {
        "segments": segments,
        "transcript": transcript,
        "unresolved": unresolved,
    }


def build_transcript(wav: np.ndarray, sr: int, turns: list[dict], stt: dict, language: str | None,
                     translate: bool, names: list[str], log, _p, t0: float) -> list[dict]:
    """Turn named speaker turns + Whisper words into finished transcript lines.

    Shared by uploaded meetings (turns from diarization + voice ID) and Phase 2
    meeting rooms (turns from each participant's own microphone). `turns` need
    start/end/speaker/name; `stt` is {"res": transcribe_words(...)} or {"err": e}.
    """
    duration = len(wav) / sr
    if "err" in stt:
        log(f"⚠ Whole-file transcription failed ({stt['err']}); transcribing turn by turn instead")
    elif not stt.get("res"):
        log("Configured model can't do whole-file mode; transcribing turn by turn instead")
    res = stt.get("res")
    words, file_lang = (res["words"], res["language"]) if res else ([], None)
    texts = _assign_words(words, turns) if res else [""] * len(turns)
    if res:
        log(f"Step 4 — Matched {len(words)} words to {len(turns)} speaker turns by timestamp")

    _p(86, "transcription")
    auto = not language or language == "auto"
    file_lang = normalize_lang(file_lang) or file_lang
    segments = []
    for t, text in zip(turns, texts):
        spk_lang = None if t["name"].startswith("Unknown") else get_speaker_language_by_name(t["name"])
        chunk = slice_audio(wav, sr, t["start"], t["end"])
        if not res:
            # No whole-file text: transcribe this turn alone (old behaviour).
            seg_lang = spk_lang or (None if auto else language)
            text = transcribe_segment(chunk, language=seg_lang)
        else:
            seg_lang = file_lang
            # Mixed-language meetings (Auto only): a speaker enrolled in another
            # language is re-transcribed in it — but only if this turn's audio
            # confirms it, since enrollment language tags can be wrong.
            if auto and spk_lang and file_lang and spk_lang != file_lang \
                    and t["end"] - t["start"] >= _TURN_ID_MIN_S:
                det, prob = detect_language_prob(chunk)
                if normalize_lang(det) == spk_lang and prob >= 0.6:
                    seg_lang = spk_lang
                    log(f"  turn at {t['start']:.1f}s ({t['name']}) is {spk_lang} ({prob:.0%} sure) "
                        f"→ re-transcribing in {spk_lang}")
                    text = transcribe_segment(chunk, language=seg_lang)
        if not text:
            continue
        segments.append({
            "start": round(t["start"], 2),
            "end": round(t["end"], 2),
            "speaker": t["name"],
            "raw_speaker": t["speaker"],
            "text": text,
            "language": seg_lang or "auto",
        })

    _p(90, "finalizing")
    segments = _merge_consecutive(segments)

    # Group segments by language once for both polish + translate.
    by_lang_pre: dict[str, list[int]] = {}
    for idx, s in enumerate(segments):
        lang_i = (s.get("language") or "").lower()
        if lang_i and lang_i != "auto":
            by_lang_pre.setdefault(lang_i, []).append(idx)

    # GPT fixes mis-heard words ("amen" -> "I'm in") using the whole conversation.
    if POLISH_TRANSCRIPT and OPENAI_API_KEY and by_lang_pre:
        _p(91, "polishing")
        for lang, indices in by_lang_pre.items():
            texts = [segments[i]["text"] for i in indices]
            try:
                with log.step(f"Step 5 — Fix mis-heard words ({len(indices)} {lang} lines, {model_label('polish')})"):
                    fixed = gpt_polish_batch(texts, lang, names=names)
                    changed = sum(1 for a, b in zip(texts, fixed) if b and a != b)
                    log(f"  {changed} of {len(indices)} lines corrected")
                for i, t in zip(indices, fixed):
                    if t:
                        segments[i]["text"] = t
            except Exception:
                log("  using the raw Whisper text instead")

    # Batch-translate AFTER merging so GPT sees full context. Group by language,
    # one API call per language, numbered lines round-trip preserved.
    if translate:
        _p(93, "translation")
        for lang, indices in by_lang_pre.items():
            if lang == "en":
                continue
            texts = [segments[i]["text"] for i in indices]
            translations: list[str] = []
            if OPENAI_API_KEY:
                try:
                    with log.step(f"Step 6 — Translate {len(indices)} {lang} lines → English "
                                  f"({model_label('translate')})"):
                        translations = gpt_translate_batch(texts, lang)
                except Exception:
                    log("  translation left blank")
                    translations = []
            for i, tr in zip(indices, translations):
                if tr:
                    segments[i]["translation"] = tr
        if not any(l != "en" for l in by_lang_pre):
            log("Step 6 — Translation skipped (everything is already English)")
    took = time.time() - t0
    who = sorted({s["speaker"] for s in segments})
    log(f"Transcript ready: {len(segments)} lines, {len(who)} speakers ({', '.join(who)}) — "
        f"{duration:.0f}s of audio in {took:.0f}s ({took / max(duration, 1):.2f}× real time)")
    return segments

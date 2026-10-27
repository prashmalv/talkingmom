"""Account-based meeting rooms: speaker names come from who joined, not voice samples.

Each participant joins from their own device under their name (their
"account") and streams their own microphone. After the meeting:

  1. every device's recording is placed on one shared timeline (clock offset
     measured at join);
  2. for each 20 ms frame the speaker is the device whose mic hears speech
     loudest *relative to its own background noise* (SNR — independent of
     how sensitive each mic is), smoothed so it doesn't flicker;
  3. a "best mic" mix is cut from whichever device is speaking, Whisper
     transcribes it once, and each word goes to the speaker of that moment.

No voice enrollment is needed. A room with a single device (everyone around
one laptop) falls back to the Phase 1 voice-recognition pipeline.
"""
from __future__ import annotations

import random
import string
import threading
import time
import traceback
import uuid

import numpy as np

from ..audio_utils import TARGET_SR, load_audio, save_wav
from ..config import BASE_DIR, OPENAI_API_KEY, UPLOAD_DIR
from ..jobs import enqueue, new_job, update_job
from ..pipeline import build_transcript, process_meeting
from ..run_log import MeetingLog
from ..transcription import transcribe_words
from . import mom as mom_store

ROOM_DIR = BASE_DIR / "data" / "phase2" / "rooms"
ROOM_DIR.mkdir(parents=True, exist_ok=True)

_ROOMS: dict[str, dict] = {}
_LOCK = threading.Lock()

_FRAME = 320           # 20 ms at 16 kHz
_ACTIVE_SNR_DB = 10.0  # a mic counts as "hearing speech" above its noise floor by this much
_SMOOTH_FRAMES = 25    # 0.5 s majority vote against flicker
_MIN_TURN_S = 0.4


def _code() -> str:
    while True:
        c = "".join(random.choices(string.ascii_uppercase.replace("O", "").replace("I", ""), k=5))
        if c not in _ROOMS:
            return c


def public(room: dict) -> dict:
    """Room state for the UI (no file paths)."""
    now = time.time()
    out = {
        "code": room["code"], "title": room["title"], "state": room["state"],
        "source": room.get("source", "devices"),
        "language": room["language"], "translate": room["translate"],
        "elapsed_s": round((room.get("ended_at") or now) - room["started_at"]) if room.get("started_at") else 0,
        "participants": [
            {"pid": p["pid"], "name": p["name"], "recording": p["track_start"] is not None and not p["final"],
             "uploaded_s": round(p["bytes"] / 4000), "online": now - p["last_seen"] < 8, "final": p["final"]}
            for p in room["participants"].values()
        ],
        "asks": [{k: a[k] for k in ("t", "question", "title", "answer")} for a in room["asks"]],
        "job_id": room.get("job_id"), "mom_id": room.get("mom_id"), "error": room.get("error"),
    }
    if room.get("bot"):
        out["bot"] = room["bot"].status()
        out["meet_url"] = room["meet_url"]
    return out


def create(title: str, language: str = "auto", translate: bool = True, meet_url: str | None = None) -> dict:
    with _LOCK:
        code = _code()
        room = {"code": code, "title": title.strip() or "Review meeting", "created_at": time.time(),
                "state": "lobby", "started_at": None, "ended_at": None,
                "language": language, "translate": translate, "source": "meet" if meet_url else "devices",
                "participants": {}, "asks": [], "job_id": None, "mom_id": None, "result": None}
        (ROOM_DIR / code).mkdir(parents=True, exist_ok=True)
        _ROOMS[code] = room
    print(f"  [room {code}] created — '{room['title']}'" + (f" (Google Meet bot → {meet_url})" if meet_url else ""),
          flush=True)
    if meet_url:
        _start_bot(room, meet_url)
    return room


# ------------------------------------------------------------ Google Meet bot

def _start_bot(room: dict, url: str) -> None:
    from .meet_bot import MeetBot, new_audio_path

    def on_event(state: str, _detail: str) -> None:
        if state == "recording" and room["state"] == "lobby":
            room.update(state="live", started_at=time.time())
        elif state in ("left", "ended") and room["state"] in ("live", "ending"):
            if room["state"] == "live":
                room["ended_at"] = time.time()
            room["state"] = "ending"
            _queue(room, _process_meet)
        elif state == "error" and room["state"] in ("lobby", "live"):
            room.update(state="error", error=room["bot"].error)

    room["meet_url"] = url
    room["bot"] = MeetBot(room["code"], url, new_audio_path(ROOM_DIR / room["code"]), on_event)
    room["bot"].start()


def _queue(room: dict, fn) -> None:
    jid = new_job()
    room.update(state="processing", job_id=jid)
    update_job(jid, progress=8, room=room["code"])
    enqueue(jid, lambda: fn(room, jid), f"room {room['code']}")


def get(code: str) -> dict | None:
    return _ROOMS.get((code or "").upper())


def join(code: str, name: str) -> dict:
    room = get(code)
    if not room:
        raise KeyError("room not found")
    if room["state"] not in ("lobby", "live"):
        raise ValueError("this meeting has already ended")
    name = name.strip()[:60]
    if not name:
        raise ValueError("name is required")
    with _LOCK:
        taken = {p["name"].lower() for p in room["participants"].values()}
        base, n = name, 2
        while name.lower() in taken:  # two people with the same name stay distinct
            name, n = f"{base} ({n})", n + 1
        pid = uuid.uuid4().hex[:10]
        room["participants"][pid] = {"pid": pid, "name": name, "joined_at": time.time(), "last_seen": time.time(),
                                     "track_start": None, "chunks": 0, "bytes": 0, "final": False, "ext": "webm"}
    print(f"  [room {room['code']}] {name} joined", flush=True)
    return {"pid": pid, "name": name, "room": public(room)}


def start(code: str) -> dict:
    room = get(code)
    if not room:
        raise KeyError("room not found")
    if room["state"] == "lobby":
        room.update(state="live", started_at=time.time())
        print(f"  [room {code}] meeting started with {len(room['participants'])} participant(s)", flush=True)
    return room


def heartbeat(code: str, pid: str) -> dict:
    room = get(code)
    if not room:
        raise KeyError("room not found")
    p = room["participants"].get(pid)
    if p:
        p["last_seen"] = time.time()
    return room


def add_audio(code: str, pid: str, seq: int, data: bytes, start_ts: float | None, final: bool, ext: str) -> None:
    room = get(code)
    if not room:
        raise KeyError("room not found")
    p = room["participants"].get(pid)
    if not p:
        raise KeyError("participant not found")
    if seq == 0:
        # Server-clock time this device started recording (client corrects its clock at join).
        p["track_start"] = start_ts or time.time()
        p["ext"] = ext if ext in ("webm", "ogg", "mp4", "m4a", "wav") else "webm"
        (ROOM_DIR / room["code"] / f"{pid}.{p['ext']}").write_bytes(b"")
    if seq != p["chunks"]:
        raise ValueError(f"chunk {seq} out of order (expected {p['chunks']})")
    with (ROOM_DIR / room["code"] / f"{pid}.{p['ext']}").open("ab") as f:
        f.write(data)
    p["chunks"] += 1
    p["bytes"] += len(data)
    p["last_seen"] = time.time()
    if final:
        p["final"] = True


def add_ask(code: str, result: dict) -> None:
    room = get(code)
    if room and room["state"] == "live" and result.get("ok"):
        room["asks"].append({"t": time.time() - room["started_at"], "question": result["question"],
                             "title": result["view"]["title"], "answer": result["answer"]})


def end(code: str) -> dict:
    """Stop the meeting; devices upload their last chunk, then processing is queued."""
    room = get(code)
    if not room:
        raise KeyError("room not found")
    if room.get("bot"):
        # Google Meet room: the bot leaves the call, then queues processing itself.
        if room["state"] == "live":
            room.update(state="ending", ended_at=time.time())
        room["bot"].stop()
        return room
    if room["state"] in ("lobby", "live"):
        room.update(state="ending", ended_at=time.time())
        print(f"  [room {code}] ended after {room['ended_at'] - (room['started_at'] or room['ended_at']):.0f}s "
              f"— waiting for final uploads", flush=True)
        threading.Thread(target=_await_uploads_then_queue, args=(room,), daemon=True).start()
    return room


def _await_uploads_then_queue(room: dict) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        recorders = [p for p in room["participants"].values() if p["chunks"]]
        if recorders and all(p["final"] for p in recorders):
            break
        time.sleep(0.5)
    jid = new_job()
    room.update(state="processing", job_id=jid)
    update_job(jid, progress=8, room=room["code"])
    enqueue(jid, lambda: _process(room, jid), f"room {room['code']}")


# ------------------------------------------------------------ processing

def _frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // _FRAME
    f = x[: n * _FRAME].reshape(n, _FRAME)
    return 10 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def _speaker_timeline(tracks: np.ndarray) -> np.ndarray:
    """Per 20 ms frame: index of the speaking device, or -1 for silence."""
    db = np.stack([_frame_db(t) for t in tracks])                 # (P, frames)
    noise = np.percentile(db, 15, axis=1, keepdims=True)
    snr = db - noise
    k = np.ones(8) / 8                                            # 160 ms smoothing
    snr = np.stack([np.convolve(s, k, mode="same") for s in snr])
    dom = np.where(snr.max(axis=0) > _ACTIVE_SNR_DB, snr.argmax(axis=0), -1)
    # Majority vote over 0.5 s windows so the label doesn't flicker on breaths.
    out = dom.copy()
    half = _SMOOTH_FRAMES // 2
    for i in np.flatnonzero(dom >= 0):
        w = dom[max(0, i - half): i + half + 1]
        w = w[w >= 0]
        out[i] = np.bincount(w).argmax()
    return out


def _turns(timeline: np.ndarray, pids: list[str], names: dict[str, str]) -> list[dict]:
    fs = _FRAME / TARGET_SR
    turns: list[dict] = []
    i, n = 0, len(timeline)
    while i < n:
        if timeline[i] < 0:
            i += 1
            continue
        j = i
        while j < n and timeline[j] == timeline[i]:
            j += 1
        spk = int(timeline[i])
        start, end = i * fs, j * fs
        if turns and turns[-1]["_idx"] == spk and start - turns[-1]["end"] < 0.6:
            turns[-1]["end"] = end  # same person after a short pause
        else:
            turns.append({"_idx": spk, "start": start, "end": end})
        i = j
    turns = [t for t in turns if t["end"] - t["start"] >= _MIN_TURN_S]
    for t in turns:
        pid = pids[t.pop("_idx")]
        t.update(speaker=pid, name=names[pid])
    return turns


def _process(room: dict, jid: str) -> None:
    code = room["code"]
    log = MeetingLog(jid[:8], on_line=lambda m: update_job(jid, detail=m.strip()))
    update_job(jid, phase="loading", started_at=time.time(), detail="Starting")
    log.banner(f"Meeting room {code} — '{room['title']}' (account-based speakers)")

    def _p(pct, phase=None):
        upd = {"progress": int(pct)}
        if phase:
            upd["phase"] = phase
        update_job(jid, **upd)

    try:
        recs = [p for p in room["participants"].values() if p["chunks"]]
        if not recs:
            raise ValueError("no audio was received from any participant")
        names = {p["pid"]: p["name"] for p in recs}
        log(f"Participants with audio: {', '.join(names.values())}")

        if len(recs) == 1:
            # One device for the whole room: fall back to voice recognition.
            p = recs[0]
            log(f"Single device ({p['name']}) — using voice recognition to tell speakers apart")
            path = ROOM_DIR / code / f"{p['pid']}.{p['ext']}"
            wav, _ = load_audio(path)
            mix_id = f"room_{code}_{uuid.uuid4().hex[:6]}.wav"
            save_wav(UPLOAD_DIR / mix_id, wav)
            res = process_meeting(UPLOAD_DIR / mix_id, language=room["language"], translate=room["translate"],
                                  progress_cb=_p, log=log)
            segments = res["segments"]
        else:
            with log.step(f"Step 1 — Align {len(recs)} device recordings on one timeline"):
                t0 = room["started_at"] or min(p["track_start"] for p in recs)
                audio, offsets = [], []
                for p in recs:
                    w, _ = load_audio(ROOM_DIR / code / f"{p['pid']}.{p['ext']}")
                    off = max(0, int(round((p["track_start"] - t0) * TARGET_SR)))
                    audio.append(w)
                    offsets.append(off)
                    log(f"  {p['name']}: {len(w) / TARGET_SR:.1f}s, starts at +{off / TARGET_SR:.2f}s")
                total = max(o + len(w) for o, w in zip(offsets, audio))
                tracks = np.zeros((len(recs), total), dtype=np.float32)
                for i, (o, w) in enumerate(zip(offsets, audio)):
                    tracks[i, o:o + len(w)] = w
            with log.step("Step 2 — Who spoke when (loudest mic vs its own background noise)"):
                pids = [p["pid"] for p in recs]
                timeline = _speaker_timeline(tracks)
                turns = _turns(timeline, pids, names)
                talk = {names[pid]: sum(t["end"] - t["start"] for t in turns if t["speaker"] == pid) for pid in pids}
                log(f"  {len(turns)} turns · " + ", ".join(f"{n} {s:.0f}s" for n, s in talk.items()))
            # "Best mic" mix: each moment taken from the device of whoever is speaking.
            src = np.repeat(np.maximum(timeline, 0), _FRAME)[:total]
            mix = tracks.mean(axis=0)
            spoken = np.repeat(timeline >= 0, _FRAME)[:total]
            idx = np.arange(len(src))
            mix[: len(src)][spoken] = tracks[src[spoken], idx[spoken]]
            mix_id = f"room_{code}_{uuid.uuid4().hex[:6]}.wav"
            save_wav(UPLOAD_DIR / mix_id, mix)
            _p(30, "transcription")
            stt: dict = {}
            t1 = time.time()
            try:
                stt["res"] = transcribe_words(mix, room["language"], list(names.values()), log=log)
                log(f"✓ Whisper done in {time.time() - t1:.1f}s — {len(stt['res']['words'])} words")
            except Exception as e:
                stt["err"] = e
            segments = build_transcript(mix, TARGET_SR, turns, stt, room["language"], room["translate"],
                                        list(names.values()), log, _p, t1)

        transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        summary = ""
        if transcript and OPENAI_API_KEY:
            _p(94, "summary")
            with log.step("Step 7 — Formal MoM draft (agenda, decisions, action items)"):
                m = mom_store.draft(room["title"], room["started_at"] or time.time(), list(names.values()),
                                    segments, room["asks"], source={"room": code, "audio_id": mix_id})
                room["mom_id"] = m["id"]
                summary = m["summary"]
        room["result"] = {"segments": segments, "transcript": transcript, "summary": summary,
                          "audio_id": mix_id, "mom_id": room.get("mom_id")}
        room["state"] = "done"
        update_job(jid, status="done", phase="done", progress=100, result=room["result"], finished_at=time.time())
        log.banner(f"Room {code} finished in {log.elapsed():.0f}s")
    except Exception as e:
        traceback.print_exc()
        room.update(state="error", error=str(e))
        update_job(jid, status="error", error=str(e), finished_at=time.time())
        log.banner(f"Room {code} FAILED: {e}")


def _process_meet(room: dict, jid: str) -> None:
    """Google Meet recording + caption speaker names → transcript → MoM."""
    code, bot = room["code"], room["bot"]
    log = MeetingLog(jid[:8], on_line=lambda m: update_job(jid, detail=m.strip()))
    update_job(jid, phase="loading", started_at=time.time(), detail="Starting")
    log.banner(f"Google Meet room {code} — '{room['title']}' (names from Meet captions)")

    def _p(pct, phase=None):
        update_job(jid, **({"progress": int(pct)} | ({"phase": phase} if phase else {})))

    try:
        if not bot.chunks or not bot.audio_path.exists():
            raise ValueError("no audio was recorded from the meeting")
        with log.step("Step 1 — Load the meeting recording"):
            wav, _ = load_audio(bot.audio_path)
            log(f"  {len(wav) / TARGET_SR:.1f}s of audio, {len(bot.utterances)} caption turns, "
                f"speakers seen: {', '.join(sorted(bot.people)) or 'none'}")
        mix_id = f"room_{code}_{uuid.uuid4().hex[:6]}.wav"
        save_wav(UPLOAD_DIR / mix_id, wav)
        turns = bot.speaker_turns()
        if turns:
            log(f"Step 2 — Who spoke when: {len(turns)} turns from Meet's captions (account names)")
            names = sorted({t["name"] for t in turns})
            _p(30, "transcription")
            stt: dict = {}
            t1 = time.time()
            try:
                stt["res"] = transcribe_words(wav, room["language"], names, log=log)
                log(f"✓ Whisper done in {time.time() - t1:.1f}s — {len(stt['res']['words'])} words")
            except Exception as e:
                stt["err"] = e
            segments = build_transcript(wav, TARGET_SR, turns, stt, room["language"], room["translate"],
                                        names, log, _p, t1)
        else:
            # Captions never came on (or nobody spoke): fall back to voice recognition.
            log("No caption speaker names were captured — falling back to voice recognition")
            res = process_meeting(UPLOAD_DIR / mix_id, language=room["language"], translate=room["translate"],
                                  progress_cb=_p, log=log)
            segments = res["segments"]
            names = sorted({s["speaker"] for s in segments})

        transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
        summary = ""
        if transcript and OPENAI_API_KEY:
            _p(94, "summary")
            with log.step("Step 7 — Formal MoM draft (agenda, decisions, action items)"):
                m = mom_store.draft(room["title"], room["started_at"] or time.time(), names, segments,
                                    room["asks"], source={"room": code, "meet": room["meet_url"], "audio_id": mix_id})
                room["mom_id"] = m["id"]
                summary = m["summary"]
        room["result"] = {"segments": segments, "transcript": transcript, "summary": summary,
                          "audio_id": mix_id, "mom_id": room.get("mom_id")}
        room["state"] = "done"
        update_job(jid, status="done", phase="done", progress=100, result=room["result"], finished_at=time.time())
        log.banner(f"Meet room {code} finished in {log.elapsed():.0f}s")
    except Exception as e:
        traceback.print_exc()
        room.update(state="error", error=str(e))
        update_job(jid, status="error", error=str(e), finished_at=time.time())
        log.banner(f"Meet room {code} FAILED: {e}")

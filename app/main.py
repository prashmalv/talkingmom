"""FastAPI app: enrollment + meeting processing + summary."""
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from .audio_utils import load_audio, save_wav, slice_audio
from .embeddings import embed
from .config import BASE_DIR, ENROLL_DIR, MEETINGS_DIR, UPLOAD_DIR
from .embeddings import embed
from .pipeline import process_meeting
from .speaker_db import delete_speaker, enroll_speaker, get_speaker_audio_path, list_speakers
from .audio_utils import TARGET_SR
from .transcription import detect_language
from .summarizer import summarize
from .startup_check import run_startup_checks

app = FastAPI(title="MoM — Meeting Transcription Demo")


@app.on_event("startup")
def _startup_preflight():
    run_startup_checks()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(dest_dir: Path, file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    raw_path = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # Normalize to 16 kHz mono WAV — torchaudio (used by pyannote) can't read m4a.
    wav_path = raw_path.with_suffix(".wav")
    if raw_path.suffix.lower() != ".wav":
        wav, sr = load_audio(raw_path)
        save_wav(wav_path, wav, sr)
        try:
            raw_path.unlink()
        except OSError:
            pass
        return wav_path
    return raw_path


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/speakers")
def api_list_speakers():
    return {"speakers": list_speakers()}


@app.post("/api/enroll")
async def api_enroll(name: str = Form(...), audio: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    path = _save_upload(ENROLL_DIR, audio)
    try:
        wav, _ = load_audio(path)
        if len(wav) < 16000:  # 1 s minimum
            raise HTTPException(400, "Enrollment audio must be at least 1 second long.")
        emb = embed(wav)
        lang = detect_language(wav) or ""
        sid = enroll_speaker(name, emb, audio_path=str(path), language=lang)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Enrollment failed: {e}")
    return {"id": sid, "name": name, "duration_s": round(len(wav) / 16000, 2), "language": lang}


@app.delete("/api/speakers/{sid}")
def api_delete_speaker(sid: str):
    delete_speaker(sid)
    return {"deleted": sid}


@app.get("/api/speakers/{sid}/audio")
def api_speaker_audio(sid: str):
    p = get_speaker_audio_path(sid)
    if not p or not Path(p).exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/wav")


# In-memory job tracker for /api/process. Simple; not durable across restarts.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _new_job() -> str:
    jid = uuid.uuid4().hex
    with _JOBS_LOCK:
        _JOBS[jid] = {
            "status": "running",
            "progress": 0,
            "phase": "upload",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }
    return jid


def _update_job(jid: str, **updates) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(updates)


def _get_job(jid: str) -> dict | None:
    with _JOBS_LOCK:
        j = _JOBS.get(jid)
        return dict(j) if j else None


def _prune_old_jobs(max_age_s: int = 3600) -> None:
    now = time.time()
    with _JOBS_LOCK:
        for jid in list(_JOBS):
            if now - _JOBS[jid].get("started_at", now) > max_age_s:
                _JOBS.pop(jid, None)


@app.post("/api/process")
async def api_process(
    audio: UploadFile = File(...),
    do_summary: bool = Form(True),
    language: str = Form("auto"),
    translate: bool = Form(False),
):
    _prune_old_jobs()
    path = _save_upload(UPLOAD_DIR, audio)
    jid = _new_job()
    _update_job(jid, progress=8, phase="upload")

    def _worker():
        try:
            def cb(pct: int, phase: str | None = None):
                upd = {"progress": max(_JOBS[jid]["progress"], min(int(pct), 99))}
                if phase:
                    upd["phase"] = phase
                _update_job(jid, **upd)

            result = process_meeting(path, language=language, translate=translate, progress_cb=cb)
            _update_job(jid, phase="summary", progress=95)
            summary = summarize(result["transcript"]) if do_summary and result["transcript"] else ""
            payload = {**result, "summary": summary, "audio_id": path.name}
            _update_job(jid, status="done", phase="done", progress=100, result=payload)
        except Exception as e:
            traceback.print_exc()
            _update_job(jid, status="error", error=str(e))

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def api_job_status(jid: str):
    j = _get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return j


_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


@app.post("/api/enroll_from_meeting")
def api_enroll_from_meeting(payload: dict = Body(...)):
    """Enroll a new speaker using audio slices from a previously processed meeting.

    Body: { name: str, audio_id: str, ranges: [[start_s, end_s], ...] }
    """
    name = (payload.get("name") or "").strip()
    audio_id = payload.get("audio_id") or ""
    ranges = payload.get("ranges") or []
    if not name:
        raise HTTPException(400, "name is required")
    if not audio_id or not ranges:
        raise HTTPException(400, "audio_id and ranges required")

    safe = Path(audio_id).name
    src = UPLOAD_DIR / safe
    if not src.exists():
        raise HTTPException(404, "meeting audio not found")

    wav, sr = load_audio(src)
    chunks = []
    for r in ranges:
        try:
            a, b = float(r[0]), float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            chunks.append(slice_audio(wav, sr, a, b))
    merged = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if len(merged) < TARGET_SR:  # < 1s
        raise HTTPException(400, "Not enough audio for this speaker (need at least 1 second)")
    # Cap at 30 s to bound the embedding call.
    if len(merged) > 30 * sr:
        merged = merged[: 30 * sr]

    # Persist the extracted audio so it can be played back from the speakers list.
    clip_path = ENROLL_DIR / f"{uuid.uuid4().hex}.wav"
    save_wav(clip_path, merged, sr)

    try:
        emb = embed(merged)
        lang = detect_language(merged) or ""
        sid = enroll_speaker(name, emb, audio_path=str(clip_path), language=lang)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Enrollment failed: {e}")
    return {"id": sid, "name": name, "duration_s": round(len(merged) / sr, 2), "language": lang}


@app.post("/api/meetings/save")
def api_save_meeting(payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    if not payload.get("segments"):
        raise HTTPException(400, "no segments to save")
    mid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    data = {
        "id": mid,
        "name": name,
        "created_at": time.time(),
        "audio_id": payload.get("audio_id"),
        "segments": payload.get("segments", []),
        "transcript": payload.get("transcript", ""),
        "summary": payload.get("summary", ""),
    }
    (MEETINGS_DIR / f"{mid}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"id": mid, "name": name, "created_at": data["created_at"]}


@app.get("/api/meetings")
def api_list_meetings():
    items = []
    for p in MEETINGS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "id": d.get("id") or p.stem,
                "name": d.get("name", "Untitled"),
                "created_at": d.get("created_at", p.stat().st_mtime),
                "turns": len(d.get("segments", [])),
            })
        except Exception:
            continue
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"meetings": items}


@app.get("/api/meetings/{mid}/data")
def api_get_meeting(mid: str):
    if not _SAFE_ID.match(mid):
        raise HTTPException(400, "bad id")
    p = MEETINGS_DIR / f"{mid}.json"
    if not p.exists():
        raise HTTPException(404, "meeting not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.delete("/api/meetings/{mid}")
def api_delete_meeting(mid: str):
    if not _SAFE_ID.match(mid):
        raise HTTPException(400, "bad id")
    p = MEETINGS_DIR / f"{mid}.json"
    if p.exists():
        p.unlink()
    return {"deleted": mid}


@app.get("/api/meetings/{audio_id}/audio")
def api_meeting_audio(audio_id: str):
    # Prevent path traversal — only allow simple filenames from UPLOAD_DIR.
    safe = Path(audio_id).name
    p = UPLOAD_DIR / safe
    if not p.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(p, media_type="audio/wav")


# --- Static frontend ---
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

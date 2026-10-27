"""Phase 2 API (/api/p2/...)."""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile

from ..audio_utils import load_audio
from ..config import MEETINGS_DIR
from ..speaker_db import normalize_lang
from ..transcription import detect_language_prob, transcribe_segment
from . import assistant, mom, rooms
from .sample_data import DISTRICTS, SOURCE_NOTE, catalog

router = APIRouter(prefix="/api/p2")


@router.get("/time")
def server_time():
    """Devices sync their clocks to this so recordings line up."""
    return {"now": time.time()}


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # picks the outgoing interface; nothing is sent
            return s.getsockname()[0]
    except OSError:
        return ""


@router.get("/net")
def network():
    """This PC's address on the local network. The host page uses it for the
    join link/QR code: 'localhost' on a phone means the phone itself."""
    return {"lan_ip": _lan_ip()}


# ------------------------------------------------------------ meeting-room screen

_SCREEN = {"version": 0, "item": None, "room": None}
_SCREEN_LOCK = threading.Lock()


def _push_screen(item: dict, room: str | None = None) -> None:
    with _SCREEN_LOCK:
        _SCREEN.update(version=_SCREEN["version"] + 1, item=item, room=room, at=time.time())


@router.get("/screen")
def get_screen():
    return _SCREEN


@router.post("/screen")
def set_screen(payload: dict = Body(...)):
    """Show something on the room screen: {question, answer, view} or {view: null} to reset."""
    _push_screen(payload if payload.get("view") else None, payload.get("room"))
    return {"version": _SCREEN["version"]}


# ------------------------------------------------------------ assistant

def _after_ask(res: dict, room: str | None, push: bool) -> dict:
    if room:
        rooms.add_ask(room, res)
    if res.get("ok") and (push or res.get("show_on_screen")):
        _push_screen({k: res.get(k) for k in ("question", "answer", "view", "language")}, room)
        res["pushed"] = True
    return res


@router.post("/ask")
def ask(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    lang = payload.get("lang") if payload.get("lang") in ("en", "hi", "mr") else None
    return _after_ask(assistant.ask(q, lang_hint=lang), payload.get("room"), bool(payload.get("push")))


def _command_prompt(lang: str) -> str:
    if lang in ("hi", "mr"):
        return "जिले: " + ", ".join(d[1] for d in DISTRICTS) + "। जल जीवन मिशन, ODF प्लस, पानी की गुणवत्ता, शिकायतें।"
    return ("Districts: " + ", ".join(d[0] for d in DISTRICTS) +
            ". Jal Jeevan Mission, ODF Plus, water quality, grievances.")


@router.post("/ask_audio")
async def ask_audio(audio: UploadFile = File(...), lang: str = Form("auto"),
                    room: str = Form(""), push: bool = Form(False)):
    """Spoken question → Whisper (with district names as vocabulary) → assistant."""
    suffix = Path(audio.filename or "q.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        tmp = f.name
    try:
        wav, _ = load_audio(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if len(wav) < 8000:
        raise HTTPException(400, "Didn't catch that — hold the mic button a little longer.")
    t0 = time.time()
    heard = lang if lang in ("en", "hi", "mr") else None
    if not heard:
        det, _prob = detect_language_prob(wav)
        heard = normalize_lang(det) or "en"
    text = transcribe_segment(wav, language=heard, prompt=_command_prompt(heard))
    print(f"  [assistant] heard ({heard}, {time.time() - t0:.1f}s): \"{text}\"", flush=True)
    if not text:
        raise HTTPException(400, "Didn't catch that — please try again.")
    res = assistant.ask(text, lang_hint=heard)
    res["heard"] = text
    return _after_ask(res, room or None, push)


# ------------------------------------------------------------ dashboards

@router.get("/catalog")
def get_catalog():
    return catalog() | {"source": SOURCE_NOTE}


@router.get("/overview")
def overview():
    """State dashboard — fixed governed queries, no LLM involved."""
    views = [
        assistant.state_overview(),
        assistant.ranking("jjm_coverage_pct", "worst", 8),
        assistant.ranking("grv_overdue", "worst", 6, by="division"),
        assistant.trend("wq_failure_pct"),
        assistant.trend("jjm_coverage_pct"),
        assistant.anomalies(),
    ]
    for v in views:
        v.pop("facts", None)
    return {"views": views, "source": SOURCE_NOTE}


@router.get("/district/{name}")
def district(name: str):
    try:
        d = assistant._district(name)
    except ValueError:
        raise HTTPException(404, "unknown district")
    views = [
        assistant.district_status(d),
        assistant.trend("jjm_coverage_pct", [d]),
        assistant.trend("wq_failure_pct", [d]),
        assistant.trend("grv_pending", [d]),
    ]
    for v in views:
        v.pop("facts", None)
    return {"views": views, "source": SOURCE_NOTE}


# ------------------------------------------------------------ meeting rooms

def _room_or_404(code: str) -> dict:
    room = rooms.get(code)
    if not room:
        raise HTTPException(404, "Meeting room not found — check the code (rooms reset when the server restarts).")
    return room


@router.post("/rooms")
def create_room(payload: dict = Body(...)):
    """Body: {title, language, translate, meet_url?}. With meet_url, the
    'MoM Assistant' bot joins that Google Meet instead of devices joining."""
    meet_url = (payload.get("meet_url") or "").strip() or None
    if meet_url:
        from .meet_bot import valid_meet_url
        if not valid_meet_url(meet_url):
            raise HTTPException(400, "That doesn't look like a Google Meet link (https://meet.google.com/abc-defg-hij).")
    room = rooms.create(payload.get("title") or "", payload.get("language") or "auto",
                        bool(payload.get("translate", True)), meet_url=meet_url)
    return rooms.public(room)


@router.get("/meet/signin")
def meet_signin_status():
    from .meet_bot import signin_status
    return signin_status()


@router.post("/meet/signin")
def meet_signin():
    """Open Chrome on the bot's profile so a person can sign in to the bot's Google account."""
    from .meet_bot import open_signin_window
    try:
        open_signin_window()
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@router.get("/rooms/{code}")
def room_status(code: str, pid: str = ""):
    room = _room_or_404(code)
    if pid:
        rooms.heartbeat(code, pid)
    return rooms.public(room)


@router.post("/rooms/{code}/join")
def join_room(code: str, payload: dict = Body(...)):
    _room_or_404(code)
    try:
        return rooms.join(code, payload.get("name") or "")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/rooms/{code}/start")
def start_room(code: str):
    _room_or_404(code)
    return rooms.public(rooms.start(code))


@router.post("/rooms/{code}/end")
def end_room(code: str):
    _room_or_404(code)
    return rooms.public(rooms.end(code))


@router.post("/rooms/{code}/audio")
async def room_audio(code: str, request: Request, pid: str, seq: int, final: int = 0,
                     start_ts: float = 0.0, ext: str = "webm"):
    _room_or_404(code)
    data = await request.body()
    try:
        rooms.add_audio(code, pid, seq, data, start_ts or None, bool(final), ext)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "seq": seq}


@router.get("/rooms/{code}/result")
def room_result(code: str):
    room = _room_or_404(code)
    if not room.get("result"):
        raise HTTPException(409, "not ready")
    return room["result"]


# ------------------------------------------------------------ formal MoM

@router.get("/moms")
def list_moms():
    return {"moms": mom.list_all()}


@router.get("/moms/{mid}")
def get_mom(mid: str):
    m = mom.load(mid)
    if not m:
        raise HTTPException(404, "MoM not found")
    return m


@router.put("/moms/{mid}")
def update_mom(mid: str, payload: dict = Body(...)):
    try:
        return mom.update(mid, payload)
    except KeyError:
        raise HTTPException(404, "MoM not found")
    except PermissionError as e:
        raise HTTPException(409, str(e))


@router.post("/moms/{mid}/approve")
def approve_mom(mid: str, payload: dict = Body(...)):
    try:
        return mom.approve(mid, payload.get("name") or "")
    except KeyError:
        raise HTTPException(404, "MoM not found")


@router.post("/moms/from_meeting")
def mom_from_meeting(payload: dict = Body(...)):
    """Draft a formal MoM from a meeting saved in the Phase 1 MoM screen."""
    mid = (payload.get("meeting_id") or "").strip()
    p = MEETINGS_DIR / f"{mid}.json"
    if not mid or not mid.replace("_", "").isalnum() or not p.exists():
        raise HTTPException(404, "saved meeting not found")
    d = json.loads(p.read_text(encoding="utf-8"))
    segs = d.get("segments") or []
    people = list(dict.fromkeys(s["speaker"] for s in segs))
    return mom.draft(d.get("name") or "", d.get("created_at") or time.time(), people, segs,
                     source={"meeting": mid, "audio_id": d.get("audio_id")})

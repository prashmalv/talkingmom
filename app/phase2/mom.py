"""Formal Minutes of Meeting: structured draft → edit → approve.

GPT drafts the MoM as structured fields (agenda, discussion points,
decisions, action items with owner/due date) from the speaker-labelled
transcript, plus any data the meeting looked at through the assistant. The
draft is edited by a person and only becomes final when approved, as the
concept note requires ("reviewed by the user before circulation").
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime

from ..config import BASE_DIR
from ..llm_client import llm_json

MOM_DIR = BASE_DIR / "data" / "phase2" / "moms"
MOM_DIR.mkdir(parents=True, exist_ok=True)
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")

_SYS = """You prepare formal Minutes of Meeting (MoM) for a Government of Maharashtra department
(Water Supply & Sanitation) review meeting, from a speaker-labelled transcript.

Write in clear, formal English (translate Hindi/Marathi content). Be faithful: include only
what was actually said — no invented facts, numbers, owners or dates.

Return JSON with exactly these keys:
{
 "title": short meeting title,
 "summary": 2-4 sentence overview,
 "agenda": [topics discussed, in order],
 "discussion": [{"topic": str, "points": [concise points, attribute to speakers by name where useful]}],
 "decisions": [decisions actually taken],
 "action_items": [{"owner": a participant name exactly as given, or "Unassigned",
                   "task": str, "due": due date/timeframe if one was said, else "",
                   "priority": "High"|"Medium"|"Low"}],
 "data_reviewed": [{"time": "mm:ss", "item": what was looked up and the key figure}]
}
Use empty lists when there is nothing for a section."""


def _clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def draft(title: str, started_at: float, participants: list[str], segments: list[dict],
          asks: list[dict] | None = None, source: dict | None = None) -> dict:
    """Ask GPT for a structured MoM draft and save it."""
    lines = []
    for s in segments:
        text = s.get("translation") or s["text"]
        lines.append(f"[{_clock(s['start'])}] {s['speaker']}: {text}")
    looked_up = [f"[{_clock(a['t'])}] asked \"{a['question']}\" → {a.get('title', '')}: {a.get('answer', '')}"
                 for a in (asks or [])]
    user = (f"Meeting title (given): {title or '(none)'}\n"
            f"Participants: {', '.join(participants) or '(unknown)'}\n\n"
            f"TRANSCRIPT\n" + "\n".join(lines) +
            ("\n\nDATA LOOKED UP DURING THE MEETING (governed assistant, sample data)\n" + "\n".join(looked_up)
             if looked_up else ""))
    out = llm_json("summary", [{"role": "system", "content": _SYS}, {"role": "user", "content": user}])

    def _owner(raw: str) -> str:
        """Map GPT's owner to a participant (case-insensitive, first name OK)."""
        raw = (raw or "").strip().lower()
        for p in participants:
            if raw and (raw == p.lower() or raw == p.lower().split()[0] or p.lower().startswith(raw + " ")):
                return p
        return "Unassigned"

    items = []
    for it in out.get("action_items") or []:
        items.append({
            "owner": _owner(it.get("owner")),
            "task": (it.get("task") or "").strip(),
            "due": (it.get("due") or "").strip(),
            "priority": it.get("priority") if it.get("priority") in ("High", "Medium", "Low") else "Medium",
            "status": "Open",
        })
    when = datetime.fromtimestamp(started_at)
    duration = max((s["end"] for s in segments), default=0)
    mom = {
        "id": f"{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "title": title or out.get("title") or "Review meeting",
        "date": when.strftime("%d %b %Y"),
        "time": when.strftime("%I:%M %p"),
        "duration": _clock(duration),
        "chair": "",
        "participants": participants,
        "summary": out.get("summary") or "",
        "agenda": [str(a) for a in out.get("agenda") or []],
        "discussion": [{"topic": d.get("topic", ""), "points": [str(p) for p in d.get("points") or []]}
                       for d in out.get("discussion") or []],
        "decisions": [str(d) for d in out.get("decisions") or []],
        "action_items": items,
        "data_reviewed": [{"time": str(d.get("time", "")), "item": str(d.get("item", ""))}
                          for d in out.get("data_reviewed") or []],
        "status": "draft",
        "approved_by": "",
        "approved_at": None,
        "created_at": time.time(),
        "source": source or {},
        "transcript": segments,
    }
    save(mom)
    return mom


def save(mom: dict) -> None:
    (MOM_DIR / f"{mom['id']}.json").write_text(json.dumps(mom, ensure_ascii=False, indent=1), encoding="utf-8")


def load(mid: str) -> dict | None:
    if not _SAFE.match(mid):
        return None
    p = MOM_DIR / f"{mid}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_all() -> list[dict]:
    out = []
    for p in MOM_DIR.glob("*.json"):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        open_items = sum(1 for a in m.get("action_items", []) if a.get("status") != "Done")
        out.append({k: m.get(k) for k in ("id", "title", "date", "time", "status", "created_at")}
                   | {"participants": len(m.get("participants", [])), "open_actions": open_items})
    return sorted(out, key=lambda m: m["created_at"] or 0, reverse=True)


EDITABLE = ("title", "chair", "summary", "agenda", "discussion", "decisions", "action_items",
            "data_reviewed", "participants")


def update(mid: str, changes: dict) -> dict:
    mom = load(mid)
    if not mom:
        raise KeyError(mid)
    if mom["status"] == "approved" and any(k != "action_items" for k in changes):
        raise PermissionError("Approved minutes are locked; only action-item status can change.")
    for k in EDITABLE:
        if k in changes:
            mom[k] = changes[k]
    mom["updated_at"] = time.time()
    save(mom)
    return mom


def approve(mid: str, approved_by: str) -> dict:
    mom = load(mid)
    if not mom:
        raise KeyError(mid)
    mom.update(status="approved", approved_by=approved_by.strip() or "Chair", approved_at=time.time())
    save(mom)
    return mom

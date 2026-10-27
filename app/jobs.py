"""In-memory background jobs, run one at a time.

The speech models use every CPU core, so parallel jobs only slow each other
down (two Hindi meetings together took ~20× real time). Every job — an
uploaded meeting or a Phase 2 meeting room — goes through one queue.
Not durable across restarts.
"""
import queue
import threading
import time
import uuid

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_QUEUE: "queue.Queue[tuple[str, callable]]" = queue.Queue()
_PENDING: list[str] = []  # queued job ids, in order


def new_job() -> str:
    jid = uuid.uuid4().hex
    with _LOCK:
        _JOBS[jid] = {
            "status": "running",
            "progress": 0,
            "phase": "upload",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }
    return jid


def update_job(jid: str, **updates) -> None:
    with _LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(updates)


def get_job(jid: str) -> dict | None:
    with _LOCK:
        j = _JOBS.get(jid)
        return dict(j) if j else None


def prune_old_jobs(max_age_s: int = 3600) -> None:
    """Forget finished jobs after an hour. Queued/running jobs are never pruned."""
    now = time.time()
    with _LOCK:
        for jid in list(_JOBS):
            j = _JOBS[jid]
            if j["status"] in ("done", "error") and now - j.get("finished_at", now) > max_age_s:
                _JOBS.pop(jid, None)


def _refresh_queue_positions() -> None:
    with _LOCK:
        for pos, jid in enumerate(_PENDING):
            if jid in _JOBS:
                _JOBS[jid]["detail"] = (
                    "Next in line — another meeting is processing" if pos == 0
                    else f"Waiting — {pos + 1} meetings ahead"
                )


def _queue_runner() -> None:
    while True:
        jid, fn = _QUEUE.get()
        with _LOCK:
            if jid in _PENDING:
                _PENDING.remove(jid)
        _refresh_queue_positions()
        try:
            fn()
        finally:
            _QUEUE.task_done()


threading.Thread(target=_queue_runner, daemon=True, name="meeting-queue").start()


def enqueue(jid: str, fn, label: str = "") -> int:
    """Queue fn() to run after earlier jobs; returns how many are ahead."""
    update_job(jid, phase="queued")
    with _LOCK:
        _PENDING.append(jid)
        ahead = _QUEUE.unfinished_tasks
    _QUEUE.put((jid, fn))
    _refresh_queue_positions()
    if ahead:
        print(f"[queue] job {jid[:8]} ({label}) queued — {ahead} meeting(s) ahead", flush=True)
    return ahead


def job_status(jid: str) -> dict | None:
    """Job dict for the UI, with elapsed seconds since it started running."""
    j = get_job(jid)
    if not j:
        return None
    end = j.get("finished_at") or time.time()
    j["elapsed_s"] = 0 if j["phase"] == "queued" else round(end - j["started_at"])
    return j

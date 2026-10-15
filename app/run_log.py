"""Readable terminal log of what one meeting's processing is doing.

Every line is stamped with seconds since the job started, so the console
shows which model is running, what it found, and how long each step took.
"""
import threading
import time

_PRINT_LOCK = threading.Lock()  # Whisper and diarization log from two threads


class MeetingLog:
    def __init__(self, job: str = "", on_line=None):
        self.t0 = time.time()
        self.job = job
        self.on_line = on_line  # e.g. mirror each line into the UI's job status

    def elapsed(self) -> float:
        return time.time() - self.t0

    def __call__(self, msg: str) -> None:
        tag = f"{self.job} " if self.job else ""
        with _PRINT_LOCK:
            print(f"  [{tag}{self.elapsed():6.1f}s] {msg}", flush=True)
        if self.on_line:
            try:
                self.on_line(msg)
            except Exception:
                pass

    def banner(self, msg: str) -> None:
        with _PRINT_LOCK:
            print("━" * 72, flush=True)
            print(f"▶ {msg}", flush=True)
            print("━" * 72, flush=True)

    def step(self, title: str) -> "_Step":
        """`with log.step("Diarization"):` logs start and duration."""
        return _Step(self, title)


class _Step:
    def __init__(self, log: MeetingLog, title: str):
        self.log, self.title = log, title

    def __enter__(self):
        self.t = time.time()
        self.log(f"{self.title} …")
        return self

    def __exit__(self, exc_type, exc, tb):
        took = time.time() - self.t
        if exc_type:
            self.log(f"✗ {self.title} failed after {took:.1f}s: {exc}")
        else:
            self.log(f"✓ {self.title} done in {took:.1f}s")
        return False

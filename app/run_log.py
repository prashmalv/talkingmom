"""Readable terminal log of what one meeting's processing is doing.

Every line is stamped with seconds since the job started, so the console
shows which model is running, what it found, and how long each step took.
"""
import threading
import time

_PRINT_LOCK = threading.Lock()  # Whisper and diarization log from two threads


class MeetingLog:
    def __init__(self, job: str = ""):
        self.t0 = time.time()
        self.job = job

    def elapsed(self) -> float:
        return time.time() - self.t0

    def __call__(self, msg: str) -> None:
        with _PRINT_LOCK:
            print(f"  [{self.elapsed():6.1f}s] {msg}", flush=True)

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

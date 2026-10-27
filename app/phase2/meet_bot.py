"""Google Meet bot: joins a call as "MoM Assistant", records it, and reads
Meet's own live captions to learn WHO spoke WHEN (names = Meet accounts).

How it works
  * A real (visible) Chrome window is driven by Playwright — Google refuses
    headless browsers. It joins as a guest with mic/camera off; the host admits it.
  * An init script wraps RTCPeerConnection, so every incoming audio track is
    mixed (Web Audio) and recorded with MediaRecorder; 3 s WebM chunks are
    streamed to Python.
  * Meet's captions are switched on. Each caption block carries the speaker's
    display name; we poll the caption area and turn its changes into
    utterances (name, start, end).
  * After the meeting, Whisper transcribes the recording (Hindi/Marathi/English)
    and each word is given to the caption speaker of that moment. Meet's
    caption *text* is only used live — the final words come from our Whisper.

Meet's page is not an API: the selectors below are heuristics and may need
updating when Google changes Meet. Demo-grade by design (see README).
"""
from __future__ import annotations

import base64
import re
import tempfile
import threading
import time
import traceback
import uuid

from ..config import BASE_DIR

BOT_NAME = "MoM Assistant"
# Google blocks guests without an account from meetings hosted by personal
# accounts ("You can't join this video call"). So the bot uses a Chrome
# profile signed in to a Google account — the person signs in once by hand
# (open_signin_window); the bot reuses the profile.
PROFILE_DIR = BASE_DIR / "data" / "phase2" / "meet_profile" / "bot"
_SIGNED_IN_MARK = PROFILE_DIR.parent / "signed_in.txt"


def _chrome_exe() -> str | None:
    import os
    import shutil
    for p in (shutil.which("chrome"),
              os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), r"Google\Chrome\Application\chrome.exe"),
              os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), r"Google\Chrome\Application\chrome.exe"),
              os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe")):
        if p and os.path.exists(p):
            return p
    return None


def open_signin_window() -> None:
    """Open a normal (not automated — Google refuses sign-in from automated
    browsers) Chrome window on the bot's profile, at Google sign-in."""
    import subprocess
    exe = _chrome_exe()
    if not exe:
        raise RuntimeError("Google Chrome is not installed")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([exe, f"--user-data-dir={PROFILE_DIR}", "--no-first-run", "--no-default-browser-check",
                      "--new-window", "https://accounts.google.com/"])
    _SIGNED_IN_MARK.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


def signin_status() -> dict:
    return {"signed_in_once": _SIGNED_IN_MARK.exists(),
            "since": _SIGNED_IN_MARK.read_text(encoding="utf-8") if _SIGNED_IN_MARK.exists() else None}
CAPTION_LAG_S = 1.0  # captions appear ~1 s after the words are spoken

_MEET_URL = re.compile(r"^https://meet\.google\.com/[a-z]{3,4}-[a-z]{4}-[a-z]{3,4}(\?.*)?$")

# Injected before Meet's own scripts: capture every incoming audio track.
_INIT_JS = r"""
(() => {
  const Orig = window.RTCPeerConnection;
  if (!Orig || Orig.__mom) return;
  window.__momTracks = [];
  function Wrapped(...args) {
    const pc = new Orig(...args);
    pc.addEventListener('track', (e) => {
      if (e.track && e.track.kind === 'audio') {
        window.__momTracks.push(e.track);
        if (window.__momOnTrack) window.__momOnTrack(e.track);
      }
    });
    return pc;
  }
  Wrapped.prototype = Orig.prototype;
  Object.setPrototypeOf(Wrapped, Orig);
  Wrapped.__mom = true;
  window.RTCPeerConnection = Wrapped;
})();
"""

_START_REC_JS = r"""
async () => {
  // Mix every incoming WebRTC audio track (each participant's voice) with Web
  // Audio. Chrome quirk: a remote track feeds Web Audio only while it is also
  // attached to a media element — without that the recording is pure silence.
  // (Tab capture is no use here: with the fake mic/camera flags Chrome hands
  // back its own test beep instead of the tab's sound.)
  const ctx = new AudioContext();
  try { await ctx.resume(); } catch (e) {}
  const dest = ctx.createMediaStreamDestination();
  const seen = new Set();
  window.__momKeep = [];
  const add = (t) => {
    if (seen.has(t.id)) return; seen.add(t.id);
    const ms = new MediaStream([t]);
    const el = new Audio(); el.srcObject = ms; el.muted = true; el.play().catch(() => {});
    window.__momKeep.push(el);
    try { ctx.createMediaStreamSource(ms).connect(dest); } catch (e) {}
  };
  window.__momTracks.forEach(add);
  window.__momOnTrack = add;
  const stream = dest.stream;
  const how = 'call audio (' + seen.size + ' stream(s))';
  // Level meter so Python can tell real sound from silence.
  const an = ctx.createAnalyser(); an.fftSize = 2048;
  ctx.createMediaStreamSource(stream).connect(an);
  const buf = new Float32Array(an.fftSize);
  window.__momLevel = () => { an.getFloatTimeDomainData(buf); let s = 0; for (const v of buf) s += v * v; return Math.sqrt(s / buf.length); };
  const rec = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  rec.ondataavailable = (e) => {
    if (!e.data || !e.data.size) return;
    const fr = new FileReader();
    fr.onload = () => window.momChunk(fr.result.split(',')[1]);
    fr.readAsDataURL(e.data);
  };
  rec.onstop = () => setTimeout(() => window.momChunk(''), 300);
  rec.start(3000);
  window.__momRec = rec;
  window.__momCtx = ctx;
  return { t: Date.now() / 1000, tracks: window.__momTracks.length, how };
}
"""

# Caption area → [[name, text], ...]. Meet marks it as a region labelled
# "Captions"; each block has the speaker name then the text.
_FIND_CAPTION_REGION_JS = r"""
() => document.querySelector('[role="region"][aria-label*="aption" i]')
   || document.querySelector('[aria-label*="aption" i][aria-live]')
   || document.querySelector('[aria-label="Captions"]')
"""

# Caption area → [[name, text], ...] in screen order. Meet shows one block per
# speaker turn: the speaker's avatar <img>, their name, then the text. A
# wrapper around several turns contains several avatars — so a speaker block
# is the SMALLEST element holding exactly one avatar plus at least two lines
# of text. (A first version took the wrapper's first line as the name, which
# credited everyone's words to whoever spoke first.)
_READ_CAPTIONS_JS = r"""
() => {
  const region = (""" + _FIND_CAPTION_REGION_JS.strip() + r""")();
  if (!region) return null;
  const lines = (el) => (el.innerText || '').split('\n').map(s => s.trim()).filter(Boolean);
  const cands = [...region.querySelectorAll('*')].filter(el =>
    el.querySelectorAll('img').length === 1 && lines(el).length >= 2);
  const blocks = cands.filter(el => !cands.some(o => o !== el && el.contains(o)));
  let out = blocks.map(el => { const l = lines(el); return [l[0], l.slice(1).join(' ')]; })
                  .filter(([n]) => n.length <= 60);
  if (!out.length) {
    // No avatars rendered: fall back to the region's direct children.
    for (const el of region.children) {
      const l = lines(el);
      if (l.length >= 2 && l[0].length <= 60) out.push([l[0], l.slice(1).join(' ')]);
    }
  }
  return out;
}
"""

# For debugging selector changes: a compact outline of the caption area.
_CAPTION_SNAPSHOT_JS = r"""
() => {
  const region = (""" + _FIND_CAPTION_REGION_JS.strip() + r""")();
  if (!region) return 'NO CAPTION REGION FOUND';
  const walk = (el, d) => {
    if (d > 7) return '';
    const own = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').slice(0, 60);
    let s = '  '.repeat(d) + '<' + el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(' ').slice(0, 2).join('.') : '') + '>' + (own ? ' "' + own + '"' : '') + '\n';
    for (const c of el.children) s += walk(c, d + 1);
    return s;
  };
  return walk(region, 0).slice(0, 6000);
}
"""


def valid_meet_url(url: str) -> bool:
    return bool(_MEET_URL.match((url or "").strip()))


class MeetBot(threading.Thread):
    """One bot per meeting room. Thread-safe status via .status()."""

    def __init__(self, room_code: str, url: str, audio_path, on_event=None):
        super().__init__(daemon=True, name=f"meet-bot-{room_code}")
        self.room_code = room_code
        self.url = url.strip()
        self.audio_path = audio_path
        self.on_event = on_event or (lambda *_: None)
        self.state = "launching"
        self.detail = "Starting Chrome…"
        self.error: str | None = None
        self.audio_start: float | None = None
        self.chunks = 0
        self.utterances: list[dict] = []  # {name, start, end, text}
        self.people: set[str] = set()
        self._stop = threading.Event()
        self._final_chunk = threading.Event()
        self._lock = threading.Lock()
        self._caption_seen = False
        self.capture = ""
        self.level = 0.0

    # ---------------------------------------------------------- public
    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            live = [{"name": u["name"], "text": u["text"][-160:]} for u in self.utterances[-6:]]
            return {"state": self.state, "detail": self.detail, "error": self.error,
                    "people": sorted(self.people), "captions": live,
                    "recorded_s": round(self.chunks * 3), "captions_on": self._caption_seen,
                    "capture": self.capture, "level": round(self.level, 4)}

    # ---------------------------------------------------------- internals
    def _set(self, state: str, detail: str) -> None:
        with self._lock:
            self.state, self.detail = state, detail
        print(f"  [meet-bot {self.room_code}] {detail}", flush=True)
        self.on_event(state, detail)

    def _on_chunk(self, b64: str) -> None:
        if not b64:
            self._final_chunk.set()
            return
        with open(self.audio_path, "ab") as f:
            f.write(base64.b64decode(b64))
        self.chunks += 1

    def _on_captions(self, blocks: list | None) -> None:
        if not blocks:
            return
        now = time.time()
        with self._lock:
            self._caption_seen = True
            for name, text in blocks[-6:]:
                if name.lower() in ("you", BOT_NAME.lower()):
                    continue
                if len(text) < 2:
                    continue
                self.people.add(name)
                # A block grows while that person talks (and Meet may trim its
                # start). Match it to one of this person's recent turns by text
                # overlap; otherwise it's a new turn.
                key = text[:25]
                match = None
                for u in reversed(self.utterances[-8:]):
                    if u["name"] != name:
                        continue
                    if text.startswith(u["text"][:25]) or u["text"].startswith(key) or key in u["text"] \
                            or u["text"][-25:] in text:
                        match = u
                        break
                if match:
                    if text != match["text"]:
                        match["text"], match["end"] = text, now
                else:
                    self.utterances.append({"name": name, "start": now, "end": now, "text": text})
            self.utterances.sort(key=lambda u: u["start"])

    def _click_text(self, page, pattern: str, timeout: int = 1500) -> bool:
        loc = page.get_by_role("button", name=re.compile(pattern, re.I))
        try:
            loc.first.click(timeout=timeout)
            return True
        except Exception:
            return False

    def _in_call(self, page) -> bool:
        try:
            return page.get_by_role("button", name=re.compile(r"leave call", re.I)).count() > 0
        except Exception:  # page mid-navigation
            return False

    @staticmethod
    def _body(page) -> str:
        try:
            return page.inner_text("body")[:2000].lower().replace("’", "'")
        except Exception:  # page mid-navigation
            return ""

    def run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.error = "Playwright is not installed (pip install playwright)"
            self._set("error", self.error)
            return
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with sync_playwright() as p:
                # Signed-in bot profile (see open_signin_window). Fake mic/camera
                # stop Meet's "allow microphone?" prompts; both are switched off
                # before joining — the bot only listens.
                ctx = p.chromium.launch_persistent_context(
                    str(PROFILE_DIR), channel="chrome", headless=False,
                    args=["--disable-blink-features=AutomationControlled",
                          "--autoplay-policy=no-user-gesture-required",
                          "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                          "--window-size=900,640"],
                    ignore_default_args=["--enable-automation"],
                    permissions=["microphone", "camera"],
                )
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                page.add_init_script(_INIT_JS)
                page.expose_function("momChunk", self._on_chunk)
                self._join(page)
                if self.state == "error":
                    ctx.close()
                    return
                self._record(page)
                ctx.close()
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            if "existing browser session" in msg or "has been closed" in msg:
                self.error = ("The bot's Chrome profile is already open — close the bot's Chrome window "
                              "(or the sign-in window) and try again.")
            else:
                self.error = msg.splitlines()[0][:200]
            self._set("error", f"Bot failed: {self.error}")

    def _join(self, page) -> None:
        self._set("joining", "Opening the Meet link…")
        page.goto(self.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        body = self._body(page)
        if "check your meeting code" in body or "meeting code has expired" in body:
            self.error = "Meet says this meeting code is wrong or has expired — check the link."
            self._set("error", self.error)
            return
        self._click_text(page, r"^got it$", 800)
        try:
            page.locator('input[aria-label="Your name"]').fill(BOT_NAME, timeout=8000)
        except Exception:
            pass  # signed-in profile: no name box
        # Mic and camera off before joining (fake devices — never send anything).
        for label in ("Turn off microphone", "Turn off camera"):
            self._click_text(page, "^" + label, 1500)
        joined = False
        for _ in range(4):
            if self._click_text(page, r"^(ask to join|join now|join)$", 4000):
                joined = True
                break
            page.wait_for_timeout(1500)
        if not joined:
            self.error = "Couldn't find Meet's “Ask to join” button."
            self._set("error", self.error)
            return
        self._set("waiting_admit", "Asked to join — the host must click Admit for “MoM Assistant” in Meet")
        deadline = time.time() + 300
        while time.time() < deadline and not self._stop.is_set():
            if self._in_call(page):
                return
            body = self._body(page)
            # Only Meet's own refusal messages count (the waiting page itself
            # mentions permissions, so loose words like "denied" mislead).
            if "can't join this video call" in body:
                self.error = ("Meet blocked the bot (“You can't join this video call”). Meetings hosted by "
                              "personal Google accounts don't admit guests without an account — sign the bot "
                              "in to a Google account first (“Sign in bot account”).")
                self._set("error", self.error)
                return
            if any(s in body for s in ("denied your request",
                                       "request to join was denied", "you've been removed",
                                       "no one responded to your request")):
                said = " ".join(body.split())[:160]
                self.error = f"Meet didn't let the bot in. Meet says: “{said}”"
                self._set("error", self.error)
                return
            page.wait_for_timeout(1000)
        if self._stop.is_set():
            self.error = "Cancelled before the bot was admitted."
        else:
            self.error = "Not admitted within 5 minutes — ask the host to click Admit in Meet."
        self._set("error", self.error)

    def _record(self, page) -> None:
        self._set("in_meeting", "In the meeting — turning on captions and recording")
        page.wait_for_timeout(1500)
        # Captions: try the button, then Meet's keyboard shortcut "c".
        if not self._click_text(page, r"turn on captions"):
            try:
                page.keyboard.press("c")
            except Exception:
                pass
        res = page.evaluate(_START_REC_JS)
        self.audio_start = res["t"]
        self.capture = res["how"]
        self._set("recording", f"Recording via {res['how']} — names from Meet captions")
        empty_since = None
        heard_sound = False
        last_level_check = time.time()
        debug_file = self.audio_path.parent / "captions_debug.txt"
        last_snapshot = 0.0
        while not self._stop.is_set():
            page.wait_for_timeout(400)
            try:
                blocks = page.evaluate(_READ_CAPTIONS_JS)
                self._on_captions(blocks)
            except Exception:
                blocks = None
            # Every 15 s while captions show: save what the caption area looks
            # like, so a Meet layout change can be diagnosed after the meeting.
            if time.time() - last_snapshot > 15:
                last_snapshot = time.time()
                try:
                    outline = page.evaluate(_CAPTION_SNAPSHOT_JS)
                    with open(debug_file, "a", encoding="utf-8") as f:
                        f.write(f"===== {time.strftime('%H:%M:%S')} parsed: {blocks}\n{outline}\n")
                except Exception:
                    pass
            if time.time() - last_level_check > 2:
                last_level_check = time.time()
                try:
                    level = float(page.evaluate("() => window.__momLevel ? window.__momLevel() : 0"))
                except Exception:
                    level = 0.0
                self.level = level
                if level > 0.002:
                    heard_sound = True
                elif not heard_sound and self._caption_seen:
                    # People are talking (captions) but the recording is silent.
                    with self._lock:
                        self.detail = ("⚠ Captions are coming in but the recording is silent — "
                                       f"capture method: {self.capture}")
            if not self._in_call(page):
                if empty_since is None:
                    empty_since = time.time()
                elif time.time() - empty_since > 5:
                    self._set("ended", "The meeting ended or the bot was removed")
                    break
            else:
                empty_since = None
            if not self._caption_seen and self.chunks and self.chunks % 5 == 0:
                # Captions still off: try again (button text differs by language/UI version).
                if not self._click_text(page, r"captions", 800):
                    page.keyboard.press("c")
        try:
            page.evaluate("() => window.__momRec && window.__momRec.state !== 'inactive' && window.__momRec.stop()")
            for _ in range(20):
                if self._final_chunk.is_set():
                    break
                page.wait_for_timeout(250)
        except Exception:
            pass
        if self._in_call(page):
            self._click_text(page, r"leave call", 2000)
        self._set("left", f"Left the meeting — {self.chunks * 3}s recorded, {len(self.utterances)} caption turns")

    # ---------------------------------------------------------- results
    def speaker_turns(self) -> list[dict]:
        """Caption utterances → turns on the recording's timeline (seconds)."""
        if self.audio_start is None:
            return []
        turns = []
        with self._lock:
            utts = list(self.utterances)
        for i, u in enumerate(utts):
            start = u["start"] - self.audio_start - CAPTION_LAG_S - 0.8  # first words precede the caption
            end = u["end"] - self.audio_start - CAPTION_LAG_S + 0.4
            if i + 1 < len(utts):
                end = min(end, utts[i + 1]["start"] - self.audio_start - CAPTION_LAG_S - 0.8)
            if end > start:
                turns.append({"start": max(0.0, start), "end": end, "speaker": u["name"], "name": u["name"]})
        return turns


def new_audio_path(room_dir) -> "Path":
    return room_dir / f"meet_{uuid.uuid4().hex[:6]}.webm"

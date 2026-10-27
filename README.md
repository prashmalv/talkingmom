# MoM — Meeting Intelligence

Upload or record a meeting and get a **speaker-labelled transcript**, an optional **English translation** (for Hindi / Marathi), and **LLM-written minutes** (Summary, Key Points, Decisions, Action Items).

Each participant enrolls their voice once; after that the app recognises who said what automatically.

---

## Features

- **Voice enrollment** — upload a 5–30 s sample or record one in the browser
- **Automatic speaker recognition** — transcript lines are labelled with enrolled names
- **Multilingual** — English, Hindi, Marathi (Devanagari), with side-by-side English translation
- **Per-speaker language** — each enrolled voice gets an auto-detected language (EN / HI / MR)
- **Upload or live record** a meeting straight from the browser mic
- **Click ▶ on any line** to replay that moment of the meeting
- **Name unknown speakers** from the transcript — their voice is enrolled from that meeting
- **Save meetings** to a sidebar, **export PDF**, dark / light theme, live progress bar

---

## How it works

```
Meeting audio → normalise to 16 kHz mono WAV (librosa / ffmpeg)
  ├─ in parallel ─ faster-whisper, ONE pass over the whole file, word timestamps
  │                (English → `small`, Hindi/Marathi → `large-v3`; names given as vocabulary)
  └─ pyannote 3.1 diarization → speaking turns (when someone starts/stops)
       → overlaps removed (each moment belongs to one speaker)
       → SpeechBrain ECAPA voiceprint per turn → turns GROUPED by voice
       → each group's pooled voice vs ChromaDB → enrolled name (or "Unknown")
  → each word goes to the speaker turn it was spoken in
  → GPT polish      (optional)  fix mis-heard words, all languages
  → GPT translate   (optional)  Hindi/Marathi → natural English
  → GPT summary     (optional)  meeting minutes
```

Diarization, speaker embeddings and Whisper run **locally**. Only the polish / translate / summary steps call OpenAI or Azure OpenAI.

**Why it's built this way**
- **One Whisper pass, not one per turn.** Whisper pads every call to a 30 s window, so transcribing 2 s turns one at a time costs ~15× more. A single pass also gives Whisper the conversation's context, which improves word accuracy.
- **Speakers are grouped by our own voiceprints, not pyannote's labels.** pyannote finds turn boundaries well, but its grouping fails on short back-and-forth conversations (it put two people under one label, or one person under several). Each turn ≥ 1 s gets an ECAPA voiceprint; turns are clustered (average linkage, `SPEAKER_CLUSTER_SIMILARITY`), and short replies join the most similar group.
- **Each group is named from all its audio together.** Pooled audio matches far more reliably than single turns. A group gets a name if it's within `SPEAKER_MATCH_THRESHOLD`, or within `SPEAKER_MATCH_LOOSE` *and* clearly closer (`SPEAKER_MATCH_MARGIN`) than the next enrolled person — that covers a different mic/room than at enrollment. One person can't be given to two groups in the same meeting.
- **Mixed-language meetings (Auto).** If a speaker's enrolled language differs from the meeting's, their turns are checked with language detection and re-transcribed in their language only when the audio confirms it.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Diarization | `pyannote/speaker-diarization-3.1` |
| Speaker embeddings | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) |
| Speech-to-text | Whisper via `faster-whisper` — `small` for English, `large-v3` for Hindi / Marathi |
| Vector DB | ChromaDB (local, file-based) |
| LLM | OpenAI or Azure OpenAI (GPT-4o / GPT-4o-mini) |
| Frontend | Single `index.html` — plain HTML/CSS/JS, jsPDF (no build step) |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11** (3.10 also works) | **Not** 3.12 / 3.13 — torch / pyannote versions used here don't support them |
| **ffmpeg** on `PATH` | Needed to read MP3 / M4A / WebM |
| **8 GB+ RAM** | Whisper `large-v3` is memory-hungry |
| **~10 GB free disk** | ~2 GB packages + ~5 GB models (downloaded on first use) |
| **HuggingFace account** | Free — for the pyannote models |
| **OpenAI or Azure OpenAI key** | For polish / translation / summary |
| NVIDIA GPU (optional) | ~10× faster transcription |

---

## Setup (first time)

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install ffmpeg

- **Windows:** `winget install Gyan.FFmpeg` (then open a **new** terminal)
- **macOS:** `brew install ffmpeg`
- **Ubuntu / Debian:** `sudo apt install ffmpeg`

Check it works: `ffmpeg -version`

### 3. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # allow the activate script in this window
.\venv\Scripts\Activate.ps1
python --version                                             # must print 3.11.x
```

> If you have several Pythons installed, plain `py` / `python` may pick 3.12 or 3.13. Always use `py -3.11`.

**macOS / Linux:**

```bash
python3.11 -m venv venv
source venv/bin/activate
python --version
```

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Downloads ~2 GB (PyTorch, pyannote, Whisper, SpeechBrain). Takes 10–20 minutes.

### 5. Get a HuggingFace token

1. Sign up at https://huggingface.co
2. Create a **Read** token: https://huggingface.co/settings/tokens
3. Open both pages below while logged in and click **Agree / Accept** on each:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 6. Create your `.env`

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in:

- `HF_TOKEN` — token from step 5
- **One** LLM provider:
  - **OpenAI:** set `OPENAI_API_KEY=sk-...`
  - **Azure OpenAI:** uncomment the `AZURE_OPENAI_*` block and fill endpoint, key and deployment names

Everything else has working defaults. `.env` is git-ignored — **never commit it**.

---

## Run

```bash
# Windows — activate the venv first if it isn't already
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

python run.py
```

Open **http://localhost:8000** in Chrome or Edge.

On startup the console prints a **preflight report** checking Python, ffmpeg, folders, tokens, packages, the speaker DB and a live LLM call:

```
========================================================================
Result: 11 passed
All systems ready — process a meeting to test end-to-end.
========================================================================
```

Any ✗ line comes with a suggested fix. Stop the server with `Ctrl+C`.

VS Code users can also press **F5** (config in `.vscode/launch.json`).

> **First run is slow:** models (~5 GB) download the first time you enroll a voice / process a meeting. They are cached in `~/.cache/huggingface/` afterwards.

### Reading the terminal while a meeting processes

Each meeting prints a step-by-step log — which model runs, what it found, and how long it took (seconds since the job started on the left). The UI's once-a-second progress polling (`GET /api/jobs/…`) is hidden so it doesn't bury this.

```
▶ Meeting job c4668586 — processing 'testing1.wav'
  [   2.8s] Audio: …wav, 49.7s, language=auto, translate=on, polish=on
  [   2.8s] Step 1 — Speech-to-text (Whisper) started in background
  [   2.8s] Step 2 — Who spoke when (pyannote/speaker-diarization-3.1, window step 5.0s) …
  [   7.2s] Whisper: language detected = en (92% sure, by 'small')
  [   7.2s] Whisper: transcribing whole file with 'small' (cpu/int8, 8 threads, beam 5, language en)
  [  17.3s]   13 speaking turns, 5 distinct voices: SPEAKER_00, … SPEAKER_04
  [  17.3s] Step 3 — Group turns by voice & identify (SpeechBrain ECAPA vs ChromaDB, …) …
  [  30.1s]   voiceprints grouped 13 turns into 3 voices (pyannote had 5)
  [  41.6s]   SPEAKER_00 (13s of speech) → Dipanshu (distance 0.27)
  [  41.6s]   SPEAKER_01 (9s of speech) → Aakash (distance 0.56, weaker match but 0.28 ahead of next person)
  [  28.1s] ✓ Whisper done in 25.3s — 131 words ('small', en)
  [  31.1s] Step 4 — Matched 131 words to 13 speaker turns by timestamp
  [  31.1s] Step 5 — Fix mis-heard words (9 en lines, Azure deployment 'gpt-4.1') …
  [  34.2s] Transcript ready: 9 lines, 3 speakers (Dipanshu, Gaurav, ojaswi) — 50s of audio in 31s
  [  34.2s] Step 7 — Meeting summary (Azure deployment 'gpt-4.1') …
▶ Job c4668586 finished in 38s
```

Useful for tuning: a speaker shown as `→ Unknown (closest Gaurav 0.48)` tells you how far `SPEAKER_MATCH_THRESHOLD` would need to move.

Lines are tagged with the job id (`[c4668586  12.3s]`). The latest line is also shown under the progress ring in the browser, with the elapsed time.

**One meeting at a time.** The models use every CPU core, so meetings are processed one after another: a second upload shows *Waiting in queue — N ahead* instead of slowing both down. **Refreshing the page resumes** the meeting that was processing — no need to upload it again.

---

## Using the app

1. **Enroll voices** (left sidebar) — type a name, upload or record a clean 5–30 s sample, click **Enroll voice**. The person appears in *Enrolled voices* with a language chip.
2. **Process a meeting** — choose **Upload file** (MP3 / WAV / M4A / WebM) or **Live meeting** (record from mic; processing starts when you stop). Pick the language (Auto / English / Hindi / Marathi — a specific language is applied to everyone; choose **Auto** for mixed-language meetings), toggle *Summary* and *English translation*, click **Process meeting**.
3. **Review** — each line shows speaker, time and text (plus English translation). Click ▶ to hear that moment.
4. **Name unknown speakers** — click an *Unknown (SPEAKER_02)* label, type the real name; their voice is enrolled from this meeting and recognised in future meetings.
5. **Save / export** — **Save meeting** stores it in the sidebar; **PDF** downloads a report.

---

## Phase 2 POC — AI-Enabled Meeting Room

A separate screen at **`/phase2`** (switch with the header link; the MoM screen above is unchanged) that demonstrates the *AI-Enabled Meeting Room* from the CCC concept note:

| Concept-note use case | Where |
|---|---|
| Display status of a district / scheme | **Ask the data** — "Show the status of Pune district" |
| Retrieve operational information | **Ask the data** — "How many grievances are overdue in Nagpur division?" |
| Compare district performance | **Ask the data** — "Compare Nashik and Nagpur on tap connections" |
| Display dashboards on the room screen | **Dashboard** tab + **Room screen** (`/present`) |
| Meeting summaries · MoM · action items | **Meeting room** → **Minutes & actions** (edit → approve → print) |

> ⚠ **All scheme figures are synthetic sample data** (fixed seed, real district names). Every answer, chart and printed MoM says so. Real department systems are plugged in later by replacing `app/phase2/sample_data.py`.

### Speakers named from accounts — no voice samples
Each participant opens the **join link / QR code** on their **own phone or laptop**, types their name, and their device streams its own microphone. After the meeting every recording is placed on one timeline (clocks synced at join), and for each moment the speaker is the device whose mic hears speech loudest *relative to its own background noise*. Names therefore come from who joined. On a simulated 3-device meeting, 97% of speaking time was attributed correctly and every line had the right name. A room with only **one** device falls back to the Phase 1 voice recognition.

### Governed data assistant
Voice (mic button) or text, in English, Hindi or Marathi. GPT only **chooses** one of six fixed, read-only query types — `state_overview`, `district_status`, `compare`, `ranking`, `trend`, `anomalies` — with validated parameters; Python computes every number, then GPT phrases the result in the asker's language. It never writes SQL or invents figures, and each answer shows the query it ran. Questions asked during a live meeting are logged into the MoM under *Data reviewed*.

### Running a Phase 2 meeting
1. Other devices need **HTTPS** for the microphone, so start with:
   ```bash
   python run.py --https
   ```
   It creates a self-signed certificate (needs `openssl`, included with Git for Windows) and prints the address, e.g. `https://172.1.0.224:8443/phase2`. Each browser warns once about the certificate → **Advanced → Proceed**. All devices must be on the same network.
2. On the host PC open `/phase2` → **Create meeting room** → share the QR code / link.
3. Each participant opens it on their own device, enters their name → **Join with microphone**.
4. Host clicks **Start meeting**; open **Room screen ↗** on the TV/projector. Ask questions any time.
5. **End meeting & prepare minutes** → transcript with names → **Open minutes** → edit, **Approve**, **Print / PDF**.

Formal minutes can also be drafted from any meeting saved on the MoM screen (**Minutes & actions → Draft**).

### Google Meet bot (live, names from Meet accounts)
On **Meeting room → Google Meet bot**, paste a Meet link and click **Send bot to the meeting**. A real Chrome window ("AI Assistant") joins the call — the host clicks **Admit**. While the meeting runs the bot records what Meet plays and switches on Meet's captions, which label every utterance with the speaker's **Google account name**. Afterwards Whisper transcribes the recording and each word goes to the caption speaker of that moment; then translation, formal MoM and action items as usual. **Stop bot & prepare minutes** makes it leave (it also notices when the host ends the call for everyone).

- **One-time setup:** Google doesn't admit guests without an account into meetings hosted by personal accounts, so the bot must be signed in. Create a Google account for it (e.g. "AI Assistant"), click **Sign in bot's Google account**, sign in in the Chrome window that opens, then **close that window**. The profile is kept in `data/phase2/meet_profile/` (git-ignored).
- Needs **Google Chrome** and `playwright` (in `requirements.txt`; it drives the installed Chrome — no extra browser download). Google blocks hidden (headless) browsers, so the bot's window is visible; keep it open and the PC awake during the meeting.
- Each participant must join from **their own device and Google account** — people sharing one device appear under one name.
- **Better words:** Meet's English captions are kept as a *second opinion*. When GPT corrects the transcript it sees each Whisper line next to what Meet heard at that moment, and takes Meet's word where Whisper's makes no sense (e.g. "a cricket bowl and a bowl" → "a cricket bat and a ball"). Used for English lines only.
- **How it records:** Meet plays voices through the browser's Web Audio engine (no `<audio>` players), and its raw WebRTC tracks can be silent to scripts. The bot taps several places at once — Web Audio output, incoming call tracks, media-element streams — and every second records from whichever carries sound. Every 15 s the terminal prints an `audio check` line (`recording_from`, per-source levels, bytes received); levels stuck at `0.0000` mean the recording is silent.
- The bot saves what Meet's caption area looked like to `data/phase2/rooms/<code>/captions_debug.txt` — use it to diagnose wrong names if Google changes Meet's page.
- Demo-grade: Meet's page isn't an API, so a Meet redesign can break joining or caption reading.

Every set of minutes keeps the **full conversation** (time · speaker · what was said, plus the English translation for Hindi/Marathi). It's shown at the bottom of the minutes, **Copy** gives it as a plain script (`Name:   text`), and **Print / PDF** adds it as an *Annexure — Full conversation*.

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | **Required.** HuggingFace token for pyannote |
| `OPENAI_API_KEY` | — | OpenAI key (or Azure key if `AZURE_OPENAI_API_KEY` not set) |
| `AZURE_OPENAI_ENDPOINT` | — | Setting this switches the app to Azure OpenAI |
| `AZURE_OPENAI_API_KEY` | — | Azure key |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure API version |
| `AZURE_OPENAI_DEPLOYMENT_TRANSLATE / _POLISH / _SUMMARY` | — | Azure deployment names (can all be the same) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Summary model (OpenAI only) |
| `GPT_POLISH_MODEL` | `gpt-4o-mini` | Polish model (OpenAI only) |
| `GPT_TRANSLATE_MODEL` | `gpt-4o` | Translation model (OpenAI only) |
| `WHISPER_MODEL` | `large-v3` | Main model (used for Hindi / Marathi): `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_MODEL_EN` | empty (`.env.example`: `small`) | Model for English; in Auto mode it also detects the language. Blank = use `WHISPER_MODEL` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` for CPU, `float16` for GPU |
| `WHISPER_BEAM_SIZE` | `5` | English beam size. Higher = more accurate, slower |
| `WHISPER_BEAM_SIZE_INDIC` | `1` | Hindi/Marathi beam size; retries are also off for these. On CPU a 55 s Hindi clip took 12+ min with beam 5 + retries, ~3 min like this, same text quality |
| `WHISPER_CPU_THREADS` | `0` | CPU threads for Whisper; `0` = all cores |
| `WHISPER_MODEL_MR` / `WHISPER_MODEL_HI` | empty | Optional HuggingFace model id for Marathi / Hindi |
| `POLISH_TRANSCRIPT` | `false` | `true` = GPT fixes mis-heard words in every language, using context and enrolled names |
| `DIARIZATION_STEP` | `0.5` | pyannote window step (fraction of 10 s). `0.1` = pyannote default, ~5× slower on CPU |
| `SPEAKER_MATCH_THRESHOLD` | `0.45` | Cosine distance for a confident match; lower = stricter |
| `SPEAKER_MATCH_LOOSE` | `0.6` | Farther matches up to this distance are accepted only if clearly ahead of the runner-up |
| `SPEAKER_MATCH_MARGIN` | `0.1` | How far ahead of the next enrolled person a loose match must be |
| `SPEAKER_CLUSTER_SIMILARITY` | `0.3` | Voiceprint similarity for grouping turns as one person. Raise if two people get merged, lower if one person gets split |
| `CHROMA_DIR`, `UPLOAD_DIR`, `ENROLL_DIR`, `MEETINGS_DIR` | `./data/...` | Storage folders (auto-created) |

---

## Project structure

```
app/
  main.py            FastAPI routes, background job tracking, static frontend
  pipeline.py        End-to-end meeting processing (diarize → match → transcribe → polish → translate)
  diarization.py     pyannote wrapper
  embeddings.py      SpeechBrain ECAPA voice embeddings (+ Windows symlink workaround)
  speaker_db.py      ChromaDB storage and speaker matching
  transcription.py   faster-whisper / HF transformers STT, GPT polish + translate prompts
  summarizer.py      GPT meeting-minutes prompt
  llm_client.py      Shared OpenAI / Azure OpenAI client
  audio_utils.py     Load / slice / save audio (16 kHz mono)
  config.py          Reads .env, defines paths
  startup_check.py   Preflight health report printed at boot
  run_log.py         Step-by-step processing log printed to the terminal
  jobs.py            One-at-a-time background job queue (shared by both screens)
  phase2/            Phase 2 POC — AI meeting room
    rooms.py         Account-based meeting rooms (per-device audio → who spoke when)
    meet_bot.py      Google Meet bot (Playwright-driven Chrome: call audio + caption speaker names)
    assistant.py     Governed data assistant (question → whitelisted query → chart + answer)
    sample_data.py   SYNTHETIC district × scheme dataset (36 districts, 12 months)
    mom.py           Formal MoM: structured draft → edit → approve
    routes.py        /api/p2/... endpoints
frontend/
  index.html         MoM screen (HTML + CSS + JS in one file)
  phase2.html        Phase 2: meeting room, assistant, dashboard, minutes
  join.html          Participant page (each person's own device)
  present.html       Full-screen meeting-room display
  p2.css, p2.js      Shared Phase 2 styles and chart rendering
data/                Runtime data — auto-created, git-ignored (voices, meetings, DB, Phase 2 rooms/minutes)
certs/               Self-signed HTTPS certificate (created by --https, git-ignored)
run.py               Starts Uvicorn on http://localhost:8000 (or --https on :8443)
requirements.txt     Pinned Python dependencies
.env.example         Configuration template (copy to .env)
```

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/speakers` | List enrolled voices |
| `POST` | `/api/enroll` | Form: `name`, `audio` — enroll a voice |
| `POST` | `/api/enroll_from_meeting` | JSON: `name`, `audio_id`, `ranges` — enroll from a processed meeting |
| `DELETE` | `/api/speakers/{id}` | Remove an enrolled voice |
| `GET` | `/api/speakers/{id}/audio` | Stream enrolled voice sample |
| `POST` | `/api/process` | Form: `audio`, `do_summary`, `language`, `translate` — returns `job_id` |
| `GET` | `/api/jobs/{id}` | Poll job progress / result |
| `GET` | `/api/meetings` | List saved meetings |
| `POST` | `/api/meetings/save` | Save a processed meeting |
| `GET` | `/api/meetings/{id}/data` | Load a saved meeting |
| `DELETE` | `/api/meetings/{id}` | Delete a saved meeting |
| `GET` | `/api/meetings/{audio_id}/audio` | Stream meeting audio |

**Phase 2 (`/api/p2/...`)**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/p2/ask` | JSON: `question`, `lang?`, `room?`, `push?` — governed data question |
| `POST` | `/api/p2/ask_audio` | Form: `audio`, `lang`, `room`, `push` — spoken question |
| `GET` | `/api/p2/overview` · `/api/p2/district/{name}` · `/api/p2/catalog` | Dashboard views / available data |
| `GET` / `POST` | `/api/p2/screen` | Read / set what the room screen shows |
| `POST` | `/api/p2/rooms` | Create a meeting room → code |
| `GET` | `/api/p2/rooms/{code}` | Room status (participants, state, job) |
| `POST` | `/api/p2/rooms/{code}/join` · `/start` · `/end` | Join by name / start / end the meeting |
| `POST` | `/api/p2/rooms/{code}/audio?pid&seq&final&start_ts&ext` | Upload one recording chunk (raw body) |
| `GET` | `/api/p2/rooms/{code}/result` | Transcript + MoM id after processing |
| `GET` / `PUT` | `/api/p2/moms` · `/api/p2/moms/{id}` | List / read / edit minutes |
| `POST` | `/api/p2/moms/{id}/approve` · `/api/p2/moms/from_meeting` | Approve (locks) / draft from a saved MoM meeting |
| `GET` | `/api/p2/time` | Server clock (devices sync to it) |

Interactive docs: http://localhost:8000/docs

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded … running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first |
| `torch` DLL error / pip can't build wheels | venv is on Python 3.12+. Delete `venv`, recreate with `py -3.11 -m venv venv` |
| `HF_TOKEN is not set` / 401 from HuggingFace | Add token to `.env` and accept both pyannote model pages |
| `ffmpeg not found` / can't load audio | Install ffmpeg, open a new terminal |
| Preflight `LLM API reachable` fails with 401 | Key doesn't belong to that endpoint / resource |
| Preflight fails with 404 `DeploymentNotFound` | `AZURE_OPENAI_DEPLOYMENT_*` doesn't match a deployment name in Azure |
| First meeting is very slow | Models are downloading (~5 GB) — one-time |
| Processing is slow on CPU | Hindi/Marathi use `large-v3` and take ~3–4× the audio length on CPU (English ~0.7×). For faster Hindi set `WHISPER_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2` (~2× faster, a few more word errors that GPT polish mostly fixes). Plug in the laptop (battery mode throttles the CPU). Check `WHISPER_MODEL_EN=small` and `DIARIZATION_STEP=0.5`. For Hindi/Marathi try `WHISPER_MODEL=medium`, or use a GPU (`WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`) |
| A person shows as "Unknown" | Raise `SPEAKER_MATCH_THRESHOLD` (e.g. `0.5`) or re-enroll with a longer sample recorded on the same mic you use for meetings |
| Names get mixed up | Lower `SPEAKER_MATCH_THRESHOLD` (e.g. `0.35`) or raise `SPEAKER_MATCH_MARGIN` |
| Two people's lines under one speaker | Raise `SPEAKER_CLUSTER_SIMILARITY` (e.g. `0.4`). The terminal log shows "grouped N turns into M voices" |
| One person split into two speakers | Lower `SPEAKER_CLUSTER_SIMILARITY` (e.g. `0.2`) |
| Lines show a wrong language / script (e.g. Urdu) | Choose the meeting language explicitly instead of Auto, or re-enroll that person speaking their usual language |
| Mis-heard words | Set `POLISH_TRANSCRIPT=true`; enroll participants (their names are used as vocabulary); improve audio quality. For higher English accuracy set `WHISPER_MODEL_EN=medium` (≈2–3× slower than `small`) or `large-v3` (slowest) |
| "Failed to fetch" during processing | Don't run Uvicorn with `--reload` (file writes restart the server) |
| Mic recording doesn't work on another machine | Browsers only allow the mic on `localhost` or HTTPS |

---

## Known limitations

This is an internal demo, **not production-hardened**:

- No authentication — anyone who can reach the server has full access (it binds to `0.0.0.0`)
- Single shared speaker database (no multi-user isolation)
- Jobs are kept in memory — lost if the server restarts (the page then says so instead of waiting forever)
- Meetings are processed one at a time; with several users, later uploads wait in the queue
- Phase 2: all scheme data is **synthetic**; meeting rooms live in memory (a server restart loses an unfinished room); people sharing one device in a room are treated as one speaker (one device per person, or one device for the whole room which then uses voice recognition); no login/roles yet
- Uploaded audio is never cleaned up automatically
- PDF export uses a Latin font, so Devanagari text doesn't render correctly in the PDF
- Long meetings: polish and translation run 40 lines at a time (with the preceding lines as context), so no lines are lost; on CPU a 1-hour English meeting takes ~25–40 min to process, Hindi/Marathi ~3 h — a GPU brings both to ~5–10 min

---

## Security notes

- **Never commit `.env`** — it holds your HuggingFace and OpenAI / Azure keys. It is already in `.gitignore`.
- The `data/` folder contains **voice recordings and meeting transcripts (personal data)** and is git-ignored. Don't share it.
- If a key is ever pushed by mistake, **revoke / regenerate it immediately** — deleting the commit is not enough.

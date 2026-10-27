"""Speech-to-text pipeline.

Two backends, chosen per language:
- faster-whisper (CTranslate2) for the default model — very fast, low RAM.
- HuggingFace transformers for language-specific fine-tuned models
  (e.g. ai4bharat/*, vasista22/*) that are not CT2-compatible out of the box.

Translation to English:
- If OPENAI_API_KEY is set, we translate via GPT-4o-mini for much higher
  Marathi/Hindi quality than Whisper's built-in translate task.
- Otherwise we fall back to Whisper's translate task.
"""
from __future__ import annotations
from functools import lru_cache
import re
import numpy as np

from faster_whisper import WhisperModel

from .config import (
    GPT_POLISH_MODEL,
    GPT_TRANSLATE_MODEL,
    OPENAI_API_KEY,
    POLISH_TRANSCRIPT,
    WHISPER_BEAM_SIZE,
    WHISPER_BEAM_SIZE_INDIC,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_MODEL_EN,
    WHISPER_MODEL_HI,
    WHISPER_MODEL_MR,
)

_LANG_MODEL_OVERRIDES = {
    "en": WHISPER_MODEL_EN,
    "mr": WHISPER_MODEL_MR,
    "hi": WHISPER_MODEL_HI,
}

# Cache model IDs that failed to load so we stop re-attempting them per segment.
_FAILED_MODELS: set[str] = set()

# One-time startup log so you see the active model routing.
print("=" * 70, flush=True)
print(f"[STT config] Default model:      {WHISPER_MODEL}", flush=True)
print(f"[STT config] Device / compute:   {WHISPER_DEVICE} / {WHISPER_COMPUTE_TYPE}", flush=True)
print(f"[STT config] Beam size:          {WHISPER_BEAM_SIZE}", flush=True)
print(f"[STT config] English model:      {WHISPER_MODEL_EN or '(none — uses default)'}", flush=True)
print(f"[STT config] Marathi override:   {WHISPER_MODEL_MR or '(none — uses default)'}", flush=True)
print(f"[STT config] Hindi override:     {WHISPER_MODEL_HI or '(none — uses default)'}", flush=True)
try:
    from .llm_client import active_backend_summary, model_label
    print(f"[STT config] Translation model:  {model_label('translate') if OPENAI_API_KEY else 'none (no API key)'}", flush=True)
    print(f"[STT config] Polish transcripts: {('yes — ' + model_label('polish')) if POLISH_TRANSCRIPT and OPENAI_API_KEY else 'no'}", flush=True)
    print(f"[STT config] Summary model:      {model_label('summary') if OPENAI_API_KEY else 'none (no API key)'}", flush=True)
    print(f"[STT config] LLM backend:        {active_backend_summary()}", flush=True)
except Exception as _e:
    pass
print("=" * 70, flush=True)

# Model IDs that are known/likely to be HuggingFace-only (not CT2-converted).
_HF_ONLY_PREFIXES = ("ai4bharat/", "vasista22/", "parthiv11/", "shrimai/")

_HALLUCINATIONS = [
    r"thank(s)? you( for watching)?\.?",
    r"thanks for watching\.?",
    r"please subscribe.*",
    r"like and subscribe.*",
    r"don'?t forget to.*",
    r"subtitles? (by|from) the amara\.org community.*",
    r"subtitles? by .*",
    r"amara\.org.*",
    r"transcription by .*",
    r"मराठीमध्ये बोलत आहे\.?",
    r"सहभागी मराठीमध्ये बोलत आहे\.?",
    r"सहभागी मराठी बोलत आहे\.?",
    r"आहे बोलत आहे बोलत आहे\.?",
    r"हिंदी में बात कर रहे हैं\.?",
    r"धन्यवाद\.?$",
]
_HALLUCINATION_RE = re.compile(
    "|".join(f"({p})" for p in _HALLUCINATIONS), flags=re.IGNORECASE
)


def _clean(text: str) -> str:
    """Drop text that is *entirely* a known silence-hallucination.

    Only whole-text matches are removed, so real speech such as
    "Thank you, that helps" is kept.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if _HALLUCINATION_RE.fullmatch(text.strip(" .,!-—")):
        return ""
    return text.strip(" ,-—")


def _is_hf_only(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _HF_ONLY_PREFIXES)


# --- Backend A: faster-whisper (default) ---

@lru_cache(maxsize=4)
def _load_fw(model_id: str) -> WhisperModel:
    print(f"[STT] Loading faster-whisper model: {model_id} (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE})", flush=True)
    m = WhisperModel(model_id, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE,
                     cpu_threads=WHISPER_CPU_THREADS)
    print(f"[STT] ✓ Loaded faster-whisper: {model_id}", flush=True)
    return m


def _fw_run(wav: np.ndarray, model_id: str, language: str | None, task: str,
            prompt: str | None = None) -> str:
    model = _load_fw(model_id)
    opts = _decode_opts(language)
    opts["condition_on_previous_text"] = False  # single short turn
    segments, _info = model.transcribe(
        wav.astype(np.float32),
        language=language,
        task=task,
        best_of=WHISPER_BEAM_SIZE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200, threshold=0.5),
        initial_prompt=prompt,
        no_speech_threshold=0.7,
        suppress_blank=True,
        **opts,
    )
    return " ".join(s.text.strip() for s in segments).strip()


# --- Backend B: HuggingFace transformers (for HF-only fine-tuned models) ---

@lru_cache(maxsize=4)
def _load_hf(model_id: str):
    """Load a HuggingFace whisper-family model as an ASR pipeline."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    device = 0 if WHISPER_DEVICE == "cuda" and torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    dev_name = "cuda" if device == 0 else "cpu"
    print(f"[STT] Loading HuggingFace model: {model_id} (device={dev_name}, dtype={dtype})", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
    print(f"[STT] ✓ Loaded HuggingFace model: {model_id}", flush=True)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        device=device,
        torch_dtype=dtype,
    )


def _hf_run(wav: np.ndarray, model_id: str, language: str | None, task: str) -> str:
    pipe = _load_hf(model_id)
    generate_kwargs = {"task": task, "num_beams": WHISPER_BEAM_SIZE}
    if language:
        generate_kwargs["language"] = language
    result = pipe(wav.astype(np.float32), generate_kwargs=generate_kwargs)
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    return text.strip()


# --- Public API ---

def get_model_id(language: str | None) -> str:
    override = _LANG_MODEL_OVERRIDES.get((language or "").lower()) if language else ""
    return override or WHISPER_MODEL


def _run(wav: np.ndarray, language: str | None, task: str, prompt: str | None = None) -> str:
    if wav.size < 1600:
        return ""
    lang = language if language and language != "auto" else None
    model_id = get_model_id(lang)

    # Route: HF-only fine-tuned models take the transformers path.
    if _is_hf_only(model_id):
        # If we've already established this model can't load, skip straight to fallback.
        if model_id in _FAILED_MODELS:
            print(f"[STT] {task}: lang={lang or 'auto'} → faster-whisper [{WHISPER_MODEL}] (override {model_id} previously unavailable)", flush=True)
            text = _fw_run(wav, WHISPER_MODEL, lang, task)
            return _clean(text)
        print(f"[STT] {task}: lang={lang or 'auto'} → HuggingFace [{model_id}]", flush=True)
        try:
            text = _hf_run(wav, model_id, lang, task)
        except Exception as e:
            _FAILED_MODELS.add(model_id)
            print(f"[STT] ⚠ HuggingFace load failed for {model_id} ({str(e).splitlines()[0]}); "
                  f"disabling override this session, falling back to faster-whisper [{WHISPER_MODEL}]", flush=True)
            text = _fw_run(wav, WHISPER_MODEL, lang, task)
    else:
        print(f"[STT] {task}: lang={lang or 'auto'} → faster-whisper [{model_id}]", flush=True)
        try:
            text = _fw_run(wav, model_id, lang, task, prompt)
        except Exception as e:
            if "/" not in model_id:
                # Built-in names (large-v3, small…) aren't HuggingFace repos — no fallback exists.
                print(f"[STT] ⚠ faster-whisper failed on a segment ({e}); skipping it", flush=True)
                return ""
            print(f"[STT] ⚠ faster-whisper failed ({e}); trying HuggingFace [{model_id}]", flush=True)
            text = _hf_run(wav, model_id, lang, task)
    return _clean(text)


_PROMPT_BASE = {
    "en": ("Participants", "Hello everyone, welcome to the meeting. Okay, let's get started."),
    "hi": ("प्रतिभागी", "नमस्ते, मीटिंग में आपका स्वागत है। चलिए शुरू करते हैं।"),
    "mr": ("सहभागी", "नमस्कार, मीटिंगमध्ये आपले स्वागत आहे. चला सुरू करूया."),
}


def _vocab_prompt(names: list[str] | None, lang: str | None = None) -> str:
    # Whisper copies the style of the prompt: a punctuated sentence keeps
    # casing/punctuation on, and listing names makes it spell them correctly.
    # The prompt must be in the meeting's script — an English prompt on Hindi
    # audio makes some models answer in English.
    label, base = _PROMPT_BASE.get((lang or "en").lower(), _PROMPT_BASE["en"])
    if names:
        base = f"{label}: {', '.join(names[:30])}. " + base
    return base


def _decode_opts(lang: str | None) -> dict:
    """Decoding settings per language (see WHISPER_BEAM_SIZE_INDIC)."""
    if lang and lang != "en":
        return dict(beam_size=WHISPER_BEAM_SIZE_INDIC, temperature=0.0,
                    condition_on_previous_text=False)
    return dict(beam_size=WHISPER_BEAM_SIZE, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                condition_on_previous_text=True, compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0)


def transcribe_words(
    wav: np.ndarray,
    language: str | None = None,
    names: list[str] | None = None,
    log=print,
) -> dict | None:
    """Transcribe the whole recording in one pass with word timestamps.

    One pass is far faster than per-turn calls (every call pads to a 30 s
    window) and gives Whisper the surrounding context, which improves words.
    Returns {"words", "language", "model"}, or None if the configured model
    for this language can't do word timestamps (HF-only override) — the
    caller then falls back to per-turn transcription.
    """
    audio = wav.astype(np.float32)
    lang = language if language and language != "auto" else None
    if audio.size == 0 or float(np.abs(audio).max()) < 1e-4:
        # Silent recording: faster-whisper's language detection crashes on it.
        log("Whisper: the recording is silent — nothing to transcribe")
        return {"words": [], "language": lang or "en", "model": "(none — silent audio)"}

    def _start(model_id: str, lang_: str | None):
        # transcribe() detects the language eagerly; segments decode lazily.
        return _load_fw(model_id).transcribe(
            audio,
            language=lang_,
            task="transcribe",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200),
            initial_prompt=_vocab_prompt(names, lang_),
            no_speech_threshold=0.6,
            **_decode_opts(lang_),
        )

    if lang is None and WHISPER_MODEL_EN and not _is_hf_only(WHISPER_MODEL_EN):
        # Auto: let the fast English model detect the language. If it's
        # English, it does the whole job; otherwise hand off to the right model.
        model_id = WHISPER_MODEL_EN
        try:
            segments, info = _start(model_id, None)
        except ValueError as e:
            # faster-whisper's language detection fails when its voice-activity
            # filter finds no speech at all (e.g. a recording of a test tone).
            if "empty" not in str(e):
                raise
            log("Whisper: no speech found in the recording — nothing to transcribe")
            return {"words": [], "language": "en", "model": f"{model_id} (no speech found)"}
        log(f"Whisper: language detected = {info.language} "
            f"({info.language_probability:.0%} sure, by '{model_id}')")
        if info.language != "en":
            lang = info.language
            model_id = get_model_id(lang)
            if _is_hf_only(model_id):
                return None
            log(f"Whisper: not English → switching to '{model_id}' for {lang}")
            segments, info = _start(model_id, lang)
    else:
        model_id = get_model_id(lang)
        if _is_hf_only(model_id):
            return None
        segments, info = _start(model_id, lang)
    used = lang or info.language
    opts = _decode_opts(used)
    log(f"Whisper: transcribing whole file with '{model_id}' "
        f"({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}, {WHISPER_CPU_THREADS} threads, "
        f"beam {opts['beam_size']}, retries {'on' if isinstance(opts['temperature'], list) else 'off'}, "
        f"language {used})")

    words: list[dict] = []
    for seg in segments:
        if _clean(seg.text) == "":
            continue  # whole segment is a silence-hallucination
        for w in seg.words or []:
            if w.word.strip():
                words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
    return {"words": words, "language": lang or getattr(info, "language", None), "model": model_id}


def transcribe_segment(wav: np.ndarray, language: str | None = None, prompt: str | None = None) -> str:
    return _run(wav, language, task="transcribe", prompt=prompt)


def translate_segment(wav: np.ndarray, language: str | None = None) -> str:
    """Translate a segment to English.

    Route: if OpenAI key is available, transcribe first (in source language),
    then translate the text via GPT-4o-mini. This gives far better
    Marathi/Hindi -> English quality than Whisper's translate task.
    Falls back to Whisper's translate task if OpenAI isn't configured or errors.
    """
    src = transcribe_segment(wav, language=language)
    if not src:
        return ""
    if OPENAI_API_KEY:
        try:
            print(f"[Translate] {language or 'auto'} → English via GPT-4o-mini", flush=True)
            return _gpt_translate(src, language)
        except Exception as e:
            print(f"[Translate] ⚠ GPT translate failed ({e}); falling back to Whisper translate", flush=True)
    # Fallback: Whisper's own translate task.
    print(f"[Translate] {language or 'auto'} → English via Whisper translate task", flush=True)
    return _clean(_run(wav, language, task="translate"))


from .llm_client import llm_chat

_LANG_FULL = {"mr": "Marathi", "hi": "Hindi", "en": "English"}


_TRANSLATE_SYS = (
    "You are a professional {src} → English translator for spoken meeting transcripts.\n"
    "Your translations must convey the SAME MEANING as the source in natural, conversational English "
    "that an English speaker would actually say — not word-for-word literal.\n"
    "\n"
    "STRICT RULES — follow every one:\n"
    "1. Output ONLY the English translation. Nothing else — no preface, no quotes, no notes, no meta-commentary.\n"
    "2. NEVER refuse. NEVER say 'I cannot translate' or ask for clarification.\n"
    "   If the input is fragmentary, garbled, or auto-generated — translate what you can. "
    "   Guess intent from context when needed. An imperfect translation is always better than an apology.\n"
    "3. Translate MEANING, not words. If the {src} idiom doesn't map, use the natural English equivalent.\n"
    "4. NEVER include Devanagari characters or transliteration in the output. English only.\n"
    "5. Preserve proper nouns (people, places, deities, texts, products) — do not translate names. "
    "   Keep specific religious/philosophical/technical terms readable to an English audience "
    "   (e.g. 'Ganpati' stays 'Ganpati', 'Atharvashirsha' stays 'Atharvashirsha' — do not force awkward English equivalents).\n"
    "6. Do not summarize, expand, or editorialize. Match the length and content of the source.\n"
    "7. Preserve tone: casual speech stays casual, formal stays formal.\n"
    "\n"
    "Domain: spoken Indian meeting / podcast content. Common topics include business, technology, "
    "spirituality, family life. Speakers may switch between {src}, Hindi, and English.\n"
    "\n"
    "EXAMPLE ({src} → English):\n"
    "Input: तुमचं सर्व काहीमध्ये खूप खूप स्वागत आहे. गणपती म्हटलं की आपल्या डोळ्यासमोर अथर्वशीर्ष येतं.\n"
    "Output: A very warm welcome to all of you. When we say 'Ganpati', the Atharvashirsha immediately comes to mind.\n"
)


def _gpt_translate(text: str, language: str | None) -> str:
    src_name = _LANG_FULL.get((language or "").lower(), "the original language")
    out = llm_chat(
        "translate",
        [
            {"role": "system", "content": _TRANSLATE_SYS.format(src=src_name)},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    # Guard against a rare GPT refusal — never let those leak into the transcript.
    lower = out.lower()
    refusal_markers = (
        "i cannot", "i am unable", "i'm unable", "as an ai",
        "please provide", "as it appears", "lacks context", "incomplete or",
    )
    if any(m in lower for m in refusal_markers):
        # Return empty rather than a refusal — the UI can hide the translation line.
        return ""
    return out


_POLISH_SYS = (
    "You are an expert {src} transcript editor for meeting/podcast audio.\n"
    "You will receive numbered {src} lines produced by automatic speech recognition. "
    "They contain typical ASR errors: wrong words that sound similar, mixed scripts "
    "(e.g. Devanagari Hindi words appearing in a Marathi sentence), broken sandhi, "
    "missing punctuation, and occasional silence-hallucinations.\n"
    "\n"
    "RULES:\n"
    "1. Output the corrected {src} lines with the SAME numbering — [1], [2], ...\n"
    "2. Keep the language as {src}. Do NOT translate. Do NOT romanize.\n"
    "3. Fix obvious ASR errors using context of the surrounding lines — e.g. a reply "
    "heard as 'Amen' or 'I mean' when the speaker is agreeing to a plan is really 'I'm in'.\n"
    "4. Add natural punctuation where it was dropped.\n"
    "5. If a line looks like a hallucination (e.g. 'Thank you for watching', 'Subtitles by Amara.org'), replace with an empty [N] line.\n"
    "6. Do NOT invent content. Only fix what is clearly a recognition error.\n"
    "7. Do NOT merge, split, reorder, or drop numbered lines.\n"
    "8. Preserve proper nouns as best you can.\n"
    "\n"
    "Format each output line EXACTLY as: [N] corrected text"
)


def gpt_polish_batch(
    texts: list[str], language: str | None, names: list[str] | None = None
) -> list[str]:
    """Clean up ASR errors in a batch of same-language transcript lines via GPT."""
    if not texts or not OPENAI_API_KEY:
        return texts
    src_name = _LANG_FULL.get((language or "").lower(), "the original language")
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    system = _POLISH_SYS.format(src=src_name)
    if names:
        system += (
            "\n\nKnown participant names (spell exactly like this when a similar-"
            f"sounding word is clearly a name): {', '.join(names[:30])}"
        )
    raw = llm_chat(
        "polish",
        [
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
        temperature=0.1,
    )
    out: dict[int, str] = {}
    for line in raw.splitlines():
        m = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return [out.get(i + 1, texts[i]) for i in range(len(texts))]


def gpt_translate_batch(texts: list[str], language: str | None) -> list[str]:
    """Translate many segments in one call. Gives GPT the full meeting context and
    produces far more coherent English than per-segment calls."""
    if not texts:
        return []
    src_name = _LANG_FULL.get((language or "").lower(), "the original language")
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    sys = (
        _TRANSLATE_SYS.format(src=src_name)
        + "\n\nYou will receive numbered lines like [1] ..., [2] ...\n"
        "Return the SAME numbering, one translation per line, in the SAME order.\n"
        "Do not merge, split, reorder, or drop any line.\n"
        "Format each output line EXACTLY as: [N] english translation"
    )
    raw = llm_chat(
        "translate",
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": numbered},
        ],
        temperature=0.2,
    )
    # Parse [N] lines back into a dict, tolerant of blank lines.
    out: dict[int, str] = {}
    for line in raw.splitlines():
        m = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return [out.get(i + 1, "") for i in range(len(texts))]


def detect_language_prob(wav: np.ndarray) -> tuple[str | None, float]:
    """(language, probability) using the fastest configured model."""
    if wav.size < 1600:
        return None, 0.0
    fast = WHISPER_MODEL_EN if WHISPER_MODEL_EN and not _is_hf_only(WHISPER_MODEL_EN) else WHISPER_MODEL
    # Detection runs eagerly inside transcribe(); the segment generator is never consumed.
    _segments, info = _load_fw(fast).transcribe(
        wav.astype(np.float32),
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    return getattr(info, "language", None), float(getattr(info, "language_probability", 0.0))


def detect_language(wav: np.ndarray) -> str | None:
    try:
        return detect_language_prob(wav)[0]
    except Exception:
        return None

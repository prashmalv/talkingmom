"""LLM-based meeting summary."""
from .config import OPENAI_API_KEY
from .llm_client import llm_chat

SYSTEM_PROMPT = """You are an expert meeting-notes assistant. Given a speaker-attributed
transcript, produce concise minutes with these sections:

## Summary
2-4 sentences on what the meeting was about.

## Key Points
Bullet list of the substantive discussion points.

## Decisions
Bullet list of decisions actually made (skip section if none).

## Action Items
Bullet list of `Owner — action — (due date if mentioned)`. Skip section if none.

Attribute quotes/points to speakers by name where relevant. Be faithful to the transcript;
do not invent facts.
"""


def summarize(transcript: str) -> str:
    if not OPENAI_API_KEY:
        return (
            "_(OPENAI_API_KEY not set — skipping summary. "
            "Add it to .env to enable LLM summaries.)_"
        )
    return llm_chat(
        "summary",
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ],
        temperature=0.2,
    )

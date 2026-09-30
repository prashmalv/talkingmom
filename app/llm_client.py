"""Shared OpenAI / Azure OpenAI client.

Auto-detects whether to use platform.openai.com (regular) or Azure OpenAI
based on which env vars are set. All other code just calls `llm_chat(...)`
with a logical role and doesn't have to care which backend runs it.
"""
from __future__ import annotations
from functools import lru_cache

from .config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_POLISH,
    AZURE_OPENAI_DEPLOYMENT_SUMMARY,
    AZURE_OPENAI_DEPLOYMENT_TRANSLATE,
    AZURE_OPENAI_ENDPOINT,
    GPT_POLISH_MODEL,
    GPT_TRANSLATE_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    USE_AZURE_OPENAI,
)


@lru_cache(maxsize=1)
def _client():
    if USE_AZURE_OPENAI:
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# Map logical roles -> (Azure deployment name, OpenAI model name).
# On Azure, the `model` param must be a DEPLOYMENT NAME (not the model id).
_ROLE_MAP = {
    "translate":  (AZURE_OPENAI_DEPLOYMENT_TRANSLATE, GPT_TRANSLATE_MODEL),
    "polish":     (AZURE_OPENAI_DEPLOYMENT_POLISH,    GPT_POLISH_MODEL),
    "summary":    (AZURE_OPENAI_DEPLOYMENT_SUMMARY,   OPENAI_MODEL),
}


def _model_for(role: str) -> str:
    azure_dep, openai_model = _ROLE_MAP.get(role, ("", OPENAI_MODEL))
    if USE_AZURE_OPENAI:
        # Fall back to the summary deployment, then translate, then polish
        # if the specific one for this role isn't configured.
        for dep in (azure_dep,
                    AZURE_OPENAI_DEPLOYMENT_SUMMARY,
                    AZURE_OPENAI_DEPLOYMENT_TRANSLATE,
                    AZURE_OPENAI_DEPLOYMENT_POLISH):
            if dep:
                return dep
        raise RuntimeError(
            "Azure OpenAI is configured but no AZURE_OPENAI_DEPLOYMENT_* is set. "
            "Add AZURE_OPENAI_DEPLOYMENT_TRANSLATE (or SUMMARY/POLISH) to your .env."
        )
    return openai_model


def llm_chat(role: str, messages: list[dict], temperature: float = 0.2) -> str:
    """Send a chat completion using the correct model for the given role."""
    client = _client()
    resp = client.chat.completions.create(
        model=_model_for(role),
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def active_backend_summary() -> str:
    """One-line description of which LLM backend is active, for startup logs."""
    if USE_AZURE_OPENAI:
        return f"Azure OpenAI @ {AZURE_OPENAI_ENDPOINT} (api={AZURE_OPENAI_API_VERSION})"
    return "OpenAI (platform.openai.com)"

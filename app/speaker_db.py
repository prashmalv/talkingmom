"""ChromaDB-backed speaker profile store."""
from functools import lru_cache
from pathlib import Path
from typing import Optional
import uuid
import numpy as np
import chromadb

from .config import CHROMA_DIR, SPEAKER_COLLECTION, SPEAKER_MATCH_THRESHOLD


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    client = get_client()
    # Cosine space; we store unit-normalized embeddings so distances are in [0, 2].
    return client.get_or_create_collection(
        name=SPEAKER_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def enroll_speaker(
    name: str,
    embedding: np.ndarray,
    audio_path: str = "",
    language: str = "",
) -> str:
    coll = get_collection()
    sid = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    coll.add(
        ids=[sid],
        embeddings=[embedding.tolist()],
        metadatas=[{"name": name, "audio_path": audio_path, "language": language or ""}],
    )
    return sid


def list_speakers() -> list[dict]:
    coll = get_collection()
    data = coll.get(include=["metadatas"])
    out = []
    for i, m in zip(data["ids"], data["metadatas"]):
        ap = m.get("audio_path") or ""
        has_audio = bool(ap) and Path(ap).exists()
        out.append({
            "id": i,
            "name": m.get("name", "unknown"),
            "has_audio": has_audio,
            "language": m.get("language") or "",
        })
    return out


def get_speaker_language_by_name(name: str) -> str | None:
    """Return the enrolled language for a speaker matched by resolved name."""
    coll = get_collection()
    data = coll.get(include=["metadatas"])
    for m in data["metadatas"]:
        if m.get("name") == name:
            return m.get("language") or None
    return None


def get_speaker_audio_path(sid: str) -> Optional[str]:
    coll = get_collection()
    data = coll.get(ids=[sid], include=["metadatas"])
    if not data["ids"]:
        return None
    return data["metadatas"][0].get("audio_path") or None


def delete_speaker(sid: str) -> None:
    get_collection().delete(ids=[sid])


def match_speaker(
    embedding: np.ndarray, threshold: Optional[float] = None
) -> tuple[Optional[str], float]:
    """Return (name, distance) for the closest enrolled speaker, or (None, dist)
    if no profiles exist or the best match is farther than the threshold."""
    thr = SPEAKER_MATCH_THRESHOLD if threshold is None else threshold
    coll = get_collection()
    if coll.count() == 0:
        return None, float("inf")
    res = coll.query(
        query_embeddings=[embedding.tolist()],
        n_results=1,
        include=["metadatas", "distances"],
    )
    if not res["ids"] or not res["ids"][0]:
        return None, float("inf")
    dist = float(res["distances"][0][0])
    name = res["metadatas"][0][0].get("name", "unknown")
    if dist <= thr:
        return name, dist
    return None, dist

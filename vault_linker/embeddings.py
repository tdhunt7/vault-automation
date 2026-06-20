"""
Embedding engine: computes and caches embeddings via OpenAI or Ollama.

Produces two embedding types per note:
  - body embedding:  note content without frontmatter (what the note discusses)
  - title embedding: title + tags (what the note IS, its identity)

The body embedding captures semantic content only — raw YAML frontmatter
(dates, source filenames, etc.) is excluded to avoid noise in similarity
scores. The title embedding captures the concentrated semantic identity.
Comparing body-to-title across note pairs breaks cosine similarity's
natural symmetry to infer link directionality.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

from vault_linker.config import (
    CACHE_DIR, CACHE_FILE, TITLE_CACHE_FILE, EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)
from vault_linker.vault import Note


# ── Ollama lifecycle management ─────────────────────────────────────────────

_ollama_process: subprocess.Popen | None = None  # tracks a process we started


def _ollama_is_running() -> bool:
    """Check if Ollama is reachable."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET",
        )
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def ensure_ollama() -> None:
    """Start Ollama silently if it isn't already running.

    Launches `ollama serve` as a background process with stdout/stderr
    suppressed. Waits up to 15 seconds for the API to become reachable.
    Tracks the process so stop_ollama() can clean it up later.
    """
    global _ollama_process

    if EMBEDDING_PROVIDER != "ollama":
        return
    if _ollama_is_running():
        return

    try:
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,  # detach from terminal signals
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Ollama not found. Install it from https://ollama.com"
        )

    # Wait for the API to come up
    for _ in range(30):
        if _ollama_is_running():
            return
        time.sleep(0.5)

    raise RuntimeError(
        "Ollama started but API not reachable after 15 seconds."
    )


def stop_ollama() -> None:
    """Stop Ollama if we started it. No-op if it was already running."""
    global _ollama_process

    if _ollama_process is None:
        return

    try:
        _ollama_process.terminate()
        _ollama_process.wait(timeout=5)
    except Exception:
        try:
            _ollama_process.kill()
        except Exception:
            pass

    _ollama_process = None


def _content_hash(text: str) -> str:
    """Hash note content to detect changes."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _title_text(note: Note) -> str:
    """Build the identity string for a note: title + tags.

    Uses the human-readable title and all tags to create a
    concentrated semantic fingerprint of what this note IS.
    """
    parts = [note.title]
    if note.tags:
        parts.extend(note.tags)
    return " ".join(parts)


def _get_openai_client():
    """Initialize OpenAI client."""
    from openai import OpenAI
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY in config.py "
            "or as an environment variable."
        )
    return OpenAI(api_key=api_key)


def embed_openai(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI API. Batches automatically."""
    client = _get_openai_client()
    all_embeddings = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=OPENAI_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        if i + batch_size < len(texts):
            time.sleep(0.1)

    return all_embeddings


def _truncate_for_embedding(text: str, max_chars: int = 24_000) -> str:
    """Truncate text to stay within embedding model context limits.

    nomic-embed-text has an 8192-token context window.  At ~3-4 chars per
    token, 24 000 chars is a conservative ceiling that avoids the 500 errors
    Ollama returns when the input exceeds the model's context.  The embedding
    still captures the note's semantic gist because the opening sections
    (title, introduction, key definitions) carry the most signal.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _read_http_error(err) -> str:
    """Extract the body from an HTTPError for diagnostics."""
    try:
        body = err.read().decode("utf-8", errors="replace")
        return body[:500]  # cap so logs stay readable
    except Exception:
        return "(could not read error body)"


def embed_ollama(texts: list[str]) -> list[list[float]]:
    """Embed texts using Ollama local API.

    Uses the /api/embed endpoint with batch input for significantly
    faster throughput vs. the legacy /api/embeddings one-at-a-time approach.
    Falls back to single-request mode if batch fails (older Ollama versions).
    """
    texts = [_truncate_for_embedding(t) for t in texts]

    # ── Try batch embedding via /api/embed ────────────────────────────────
    batch_size = 10  # Keep payloads small; academic notes can be 5-10KB each
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "input": batch,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                all_embeddings.extend(result["embeddings"])
        except urllib.error.HTTPError as e:
            detail = _read_http_error(e)
            print(f"[embed] Batch /api/embed failed ({e.code}): {detail}")
            print("[embed] Falling back to single-request mode...")
            return _embed_ollama_single(texts)
        except (KeyError, urllib.error.URLError) as e:
            print(f"[embed] Batch /api/embed error: {e}")
            return _embed_ollama_single(texts)

    return all_embeddings


def _embed_ollama_single(texts: list[str]) -> list[list[float]]:
    """Fallback: embed one text at a time via legacy /api/embeddings.

    Includes retry logic with backoff to handle transient worker crashes
    (Ollama's internal model subprocess can die under sustained load,
    producing an EOF error, then respawn on the next request).
    """
    max_retries = 3
    all_embeddings = []
    for idx, text in enumerate(texts):
        text = _truncate_for_embedding(text)
        last_error = None

        for attempt in range(max_retries):
            payload = json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": text,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    all_embeddings.append(result["embedding"])
                    last_error = None
                    break
            except urllib.error.HTTPError as e:
                last_error = _read_http_error(e)
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    print(
                        f"[embed] Text {idx+1}/{len(texts)} attempt "
                        f"{attempt+1} failed ({e.code}), retrying in {wait}s..."
                    )
                    time.sleep(wait)

        if last_error is not None:
            raise RuntimeError(
                f"Ollama /api/embeddings failed on text {idx+1}/{len(texts)} "
                f"({len(text)} chars) after {max_retries} attempts: {last_error}"
            )

    return all_embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using the configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        return embed_openai(texts)
    elif EMBEDDING_PROVIDER == "ollama":
        return embed_ollama(texts)
    else:
        raise ValueError(f"Unknown provider: {EMBEDDING_PROVIDER}")


class EmbeddingCache:
    """Manages cached embeddings with content-hash invalidation."""

    def __init__(self, cache_file: Path):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_file
        self.cache: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.cache = {}

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def get_embedding(self, filename: str) -> list[float] | None:
        entry = self.cache.get(filename)
        if entry:
            return entry.get("embedding")
        return None

    def is_stale(self, filename: str, content_hash: str) -> bool:
        entry = self.cache.get(filename)
        if not entry:
            return True
        return entry.get("hash") != content_hash

    def set_embedding(self, filename: str, content_hash: str, embedding: list[float]):
        self.cache[filename] = {
            "hash": content_hash,
            "embedding": embedding,
            "provider": EMBEDDING_PROVIDER,
            "model": OPENAI_MODEL if EMBEDDING_PROVIDER == "openai" else OLLAMA_MODEL,
        }

    def save(self):
        self._save()


def unload_ollama_model() -> None:
    """Tell Ollama to unload the embedding model from memory.

    Sends keep_alive=0 so the model is freed immediately after
    indexing, rather than sitting in VRAM/RAM until Ollama's
    default timeout (5 minutes).
    """
    if EMBEDDING_PROVIDER != "ollama":
        return
    try:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "keep_alive": 0,
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # best-effort; don't crash if Ollama is unreachable


def compute_embeddings(
    notes: dict[str, Note],
    progress_callback=None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Compute body and title embeddings for all notes, using cache.

    Returns:
        (body_embeddings, title_embeddings) — both dict[filename, numpy vector]
    """
    # ── Ensure Ollama is running ────────────────────────────────────────
    needs_embedding = True  # assume yes; we'll check below
    ensure_ollama()

    body_cache = EmbeddingCache(CACHE_FILE)
    title_cache = EmbeddingCache(TITLE_CACHE_FILE)

    body_embeddings: dict[str, np.ndarray] = {}
    title_embeddings: dict[str, np.ndarray] = {}

    # ── Check what needs embedding ────────────────────────────────────────
    body_to_embed: list[str] = []
    title_to_embed: list[str] = []

    for filename, note in notes.items():
        body_hash = _content_hash(note.content)
        title_hash = _content_hash(_title_text(note))

        # Body embeddings
        if not body_cache.is_stale(filename, body_hash):
            body_embeddings[filename] = np.array(body_cache.get_embedding(filename))
        else:
            body_to_embed.append(filename)

        # Title embeddings
        if not title_cache.is_stale(filename, title_hash):
            title_embeddings[filename] = np.array(title_cache.get_embedding(filename))
        else:
            title_to_embed.append(filename)

    # ── Embed bodies ──────────────────────────────────────────────────────
    if body_to_embed:
        if progress_callback:
            progress_callback(
                f"Embedding {len(body_to_embed)} note bodies "
                f"({len(notes) - len(body_to_embed)} cached)..."
            )

        texts = [notes[fn].content for fn in body_to_embed]
        new_vectors = embed_texts(texts)

        for filename, vector in zip(body_to_embed, new_vectors):
            h = _content_hash(notes[filename].content)
            body_cache.set_embedding(filename, h, vector)
            body_embeddings[filename] = np.array(vector)

        body_cache.save()

    # ── Embed titles ──────────────────────────────────────────────────────
    if title_to_embed:
        if progress_callback:
            progress_callback(
                f"Embedding {len(title_to_embed)} note titles "
                f"({len(notes) - len(title_to_embed)} cached)..."
            )

        texts = [_title_text(notes[fn]) for fn in title_to_embed]
        new_vectors = embed_texts(texts)

        for filename, vector in zip(title_to_embed, new_vectors):
            h = _content_hash(_title_text(notes[filename]))
            title_cache.set_embedding(filename, h, vector)
            title_embeddings[filename] = np.array(vector)

        title_cache.save()

    # ── Summary ───────────────────────────────────────────────────────────
    total_new = len(body_to_embed) + len(title_to_embed)
    if total_new > 0:
        if progress_callback:
            progress_callback(f"Done. {len(notes)} notes ready ({total_new} newly embedded).")
    else:
        if progress_callback:
            progress_callback(f"All {len(notes)} notes loaded from cache.")

    # Free resources: if we started Ollama, shut it down entirely;
    # otherwise just unload the model from memory.
    if _ollama_process is not None:
        stop_ollama()
    else:
        unload_ollama_model()

    return body_embeddings, title_embeddings
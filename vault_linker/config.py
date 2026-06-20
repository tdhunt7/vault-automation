"""
Vault Linker Configuration
Edit these paths and settings to match your setup.
"""

import os
from pathlib import Path

# ── Vault ────────────────────────────────────────────────────────────────────
# Set the VAULT_PATH environment variable to override, e.g.:
#   export VAULT_PATH=~/ObsidianVaults/MyVault
VAULT_PATH = Path(os.environ.get("VAULT_PATH", str(Path.home() / "ObsidianVaults" / "MyVault")))

# Standard Obsidian internals are always excluded.
# Add your own folders via VAULT_EXCLUDE_DIRS env var (comma-separated):
#   export VAULT_EXCLUDE_DIRS="Field Notes,Inbox"
_base_exclude = {".obsidian", ".smart-env", "Templates", "Assets"}
_extra = os.environ.get("VAULT_EXCLUDE_DIRS", "")
EXCLUDE_DIRS = _base_exclude | {d.strip() for d in _extra.split(",") if d.strip()}

# ── Embeddings ───────────────────────────────────────────────────────────────
# Provider: "openai" or "ollama"
EMBEDDING_PROVIDER = "ollama"

# OpenAI settings
OPENAI_MODEL = "text-embedding-3-small"
# Set your key here or use the OPENAI_API_KEY environment variable
OPENAI_API_KEY = None  # e.g. "sk-..."

# Ollama settings (for local fallback)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Cache ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "embeddings.json"
TITLE_CACHE_FILE = CACHE_DIR / "title_embeddings.json"

# ── Display ──────────────────────────────────────────────────────────────────
MAX_CONNECTIONS = 20  # how many connections to show per note

# Similarity threshold above which the TUI shows action hints
# ("add link", "needs backlink") on unsatisfied connections.
ACTION_HINT_THRESHOLD = 0.75

# ── Tag Suggestion Confidence ────────────────────────────────────────────────
# Thresholds for color-coding suggested tags by confidence level.
#   High (green):   weight >= HIGH_WEIGHT and peers >= HIGH_PEERS
#   Moderate (yellow): weight >= MOD_WEIGHT and peers >= MOD_PEERS
#   Low (dim gray): below moderate thresholds
TAG_HIGH_WEIGHT = 1.5
TAG_HIGH_PEERS = 3
TAG_MOD_WEIGHT = 0.8
TAG_MOD_PEERS = 2

# Color mapping for folders — customize to match your vault's top-level folders.
# Using brighter variants for dark terminal backgrounds.
FOLDER_COLORS = {
    # "Folder Name": "#hex",
    "Topic A":   "#50fa7b",  # bright green
    "Topic B":   "#8be9fd",  # bright cyan-blue
    "Topic C":   "#6bc5f0",  # lighter cyan
    "Topic D":   "#ff79c6",  # bright pink
    "Topic E":   "#f1fa8c",  # bright yellow
    "Topic F":   "#ff5555",  # bright red
    "Topic G":   "#6272a4",  # muted slate
    # ── Additional color options ────────────────────────────────────────
    # "#ff9f1c",  # amber-orange
    # "#e63946",  # warm red
    # "#f4a261",  # sandy amber
    # "#e091c9",  # soft pink
    # "#6b7d8d",  # cool slate
    # "#ff6b6b",  # coral
    # "#a8d8ea",  # powder blue
    # "#c3f0ca",  # mint green
}

# ── Directionality ───────────────────────────────────────────────────────────
# Minimum z-score for asymmetric delta to declare directionality.
# Below this threshold, connection is labeled bidirectional.
DIRECTION_Z_THRESHOLD = 1.5

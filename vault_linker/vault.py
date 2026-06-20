"""
Vault parser: reads markdown files, parses YAML frontmatter,
extracts content, and finds existing [[wiki-links]].
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vault_linker.config import VAULT_PATH, EXCLUDE_DIRS


@dataclass
class Note:
    """Represents a single vault note."""
    path: Path
    filename: str          # e.g. "Python-Pandas-Overview"
    folder: str            # e.g. "Python"
    title: str             # from frontmatter or filename
    tags: list[str] = field(default_factory=list)
    created: str = ""
    source: str = ""
    note_type: str = ""    # from frontmatter 'type' field
    language: str = ""
    status: str = ""
    project: str = ""
    content: str = ""      # body text (no frontmatter)
    raw_text: str = ""     # full file text for embedding
    links: list[str] = field(default_factory=list)  # existing [[links]]

    @property
    def relative_path(self) -> str:
        return str(self.path.relative_to(VAULT_PATH))

    @property
    def key(self) -> str:
        """Unique identifier: folder/filename"""
        return f"{self.folder}/{self.filename}"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown text."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}

    body = parts[2].strip()
    return meta, body


def extract_wiki_links(text: str) -> list[str]:
    """Find all [[wiki-links]] in text, stripping aliases and headers."""
    pattern = r'\[\[([^\]|#]+)[^\]]*\]\]'
    matches = re.findall(pattern, text)
    return list(set(matches))


def extract_see_also_links(text: str) -> list[str]:
    """Find [[links]] specifically in the See Also section."""
    see_also_pattern = r'(?:^|\n)## See [Aa]lso\s*\n(.*?)(?=\n## |\Z)'
    match = re.search(see_also_pattern, text, re.DOTALL)
    if not match:
        return []
    return extract_wiki_links(match.group(1))


def load_note(filepath: Path) -> Note | None:
    """Load and parse a single markdown file into a Note."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None

    meta, body = parse_frontmatter(text)
    folder = filepath.parent.name
    filename = filepath.stem

    return Note(
        path=filepath,
        filename=filename,
        folder=folder,
        title=meta.get("title", filename),
        tags=meta.get("tags", []) or [],
        created=str(meta.get("created", "")),
        source=str(meta.get("source", "")),
        note_type=str(meta.get("type", "")),
        language=str(meta.get("language", "")),
        status=str(meta.get("status", "")),
        project=str(meta.get("project", "")),
        content=body,
        raw_text=text,
        links=extract_wiki_links(body),
    )


def load_vault() -> dict[str, Note]:
    """Load all markdown notes from the vault, keyed by filename."""
    notes: dict[str, Note] = {}

    for md_file in VAULT_PATH.rglob("*.md"):
        # Skip excluded directories
        rel_parts = md_file.relative_to(VAULT_PATH).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts):
            continue

        note = load_note(md_file)
        if note:
            notes[note.filename] = note

    return notes

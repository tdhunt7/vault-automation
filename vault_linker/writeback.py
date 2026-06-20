"""
Write-back engine: inserts [[wiki-links]] and tags into notes.

Link write-back:
  → (references):     add [[target]] to source's See Also
  ← (referenced by):  add [[source]] to target's See Also
  ↔ (related):        add [[target]] to source's See Also

Tag write-back:
  Appends a tag to the note's YAML frontmatter tags list,
  preserving existing formatting (block or flow style).

Preserves existing file content and creates sections as needed.
"""

import re
from pathlib import Path

from vault_linker.config import VAULT_PATH


def _find_see_also_range(text: str) -> tuple[int, int] | None:
    """Find the start and end positions of the See Also section.

    Returns (section_start, section_end) where section_end is
    either the start of the next ## heading or end of file.
    Returns None if no See Also section exists.
    """
    pattern = r'(?:^|\n)(## See [Aa]lso\s*\n)'
    match = re.search(pattern, text)
    if not match:
        return None

    section_start = match.start(1)
    rest = text[match.end():]

    # Find next heading or end of file
    next_heading = re.search(r'^## ', rest, re.MULTILINE)
    if next_heading:
        section_end = match.end() + next_heading.start()
    else:
        section_end = len(text)

    return section_start, section_end


def _link_already_exists(see_also_text: str, target_name: str) -> bool:
    """Check if a [[link]] to the target already exists in the section."""
    # Match [[target_name]] with optional alias or header
    pattern = rf'\[\[{re.escape(target_name)}(?:[|#][^\]]*?)?\]\]'
    return bool(re.search(pattern, see_also_text))


def add_see_also_link(filepath: Path, target_name: str) -> tuple[bool, str]:
    """Add a [[wiki-link]] to a note's See Also section.

    Creates the section if it doesn't exist. Skips if the link
    is already present.

    Args:
        filepath: Path to the markdown file to modify.
        target_name: The note name to link to (without .md extension).

    Returns:
        (success, message) tuple.
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return False, f"Cannot read {filepath.name}: {e}"

    link = f"[[{target_name}]]"
    range_result = _find_see_also_range(text)

    if range_result:
        section_start, section_end = range_result
        section_text = text[section_start:section_end]

        if _link_already_exists(section_text, target_name):
            return False, f"{link} already in {filepath.name} See Also"

        # Append link as a new list item at end of section
        # Strip trailing whitespace from section, add link, restore spacing
        section_stripped = section_text.rstrip()
        new_section = f"{section_stripped}\n- {link}\n"

        # Preserve any trailing content after the section
        new_text = text[:section_start] + new_section + text[section_end:]
    else:
        # No See Also section — create one at the end
        text_stripped = text.rstrip()
        new_text = f"{text_stripped}\n\n## See Also\n\n- {link}\n"

    try:
        filepath.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return False, f"Cannot write {filepath.name}: {e}"

    return True, f"Added {link} to {filepath.name}"


def resolve_writeback(
    source_filename: str,
    target_filename: str,
    direction: str,
) -> tuple[str, str]:
    """Determine which file to modify and what link to add.

    Args:
        source_filename: The note currently selected in the TUI.
        target_filename: The connection being acted on.
        direction: "→", "←", or "↔"

    Returns:
        (file_to_modify, link_to_add) — both are filenames without .md.
        file_to_modify is the note that gets the new See Also entry.
        link_to_add is the note name that goes inside [[brackets]].
    """
    if direction == "←":
        # Target should backlink to source
        return target_filename, source_filename
    else:
        # Source references target (→ or ↔)
        return source_filename, target_filename


def execute_writeback(
    file_to_modify: str,
    link_to_add: str,
) -> tuple[bool, str]:
    """Execute a write-back by finding the file and adding the link.

    Args:
        file_to_modify: Filename (without .md) of the note to edit.
        link_to_add: Filename (without .md) to add as [[link]].

    Returns:
        (success, message) tuple.
    """
    # Find the file in the vault
    candidates = list(VAULT_PATH.rglob(f"{file_to_modify}.md"))
    if not candidates:
        return False, f"File not found: {file_to_modify}.md"
    if len(candidates) > 1:
        return False, f"Ambiguous: multiple files named {file_to_modify}.md"

    filepath = candidates[0]
    return add_see_also_link(filepath, link_to_add)


# ── Tag write-back ──────────────────────────────────────────────────────────


def add_tag_to_frontmatter(filepath: Path, new_tag: str) -> tuple[bool, str]:
    """Add a tag to a note's YAML frontmatter tags list.

    Handles three frontmatter styles:
      - Block list:  tags:\\n  - a\\n  - b
      - Flow list:   tags: [a, b]
      - Empty/missing tags key

    Creates frontmatter if the note has none.
    Preserves existing formatting and key ordering.

    Args:
        filepath: Path to the markdown file.
        new_tag: Tag string to add.

    Returns:
        (success, message) tuple.
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return False, f"Cannot read {filepath.name}: {e}"

    if not text.startswith("---"):
        # No frontmatter — create minimal one with the tag
        new_text = f"---\ntags:\n  - {new_tag}\n---\n\n{text}"
        try:
            filepath.write_text(new_text, encoding="utf-8")
        except OSError as e:
            return False, f"Cannot write {filepath.name}: {e}"
        return True, f"Added tag '{new_tag}' to {filepath.name}"

    # Split on the closing --- to isolate the YAML block
    end_marker = text.index("---", 3)
    fm_text = text[3:end_marker]     # YAML between the --- markers
    body = text[end_marker + 3:]     # everything after closing ---

    lines = fm_text.split("\n")

    # Find the tags: line
    tags_line_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^tags\s*:', line):
            tags_line_idx = i
            break

    if tags_line_idx is None:
        # No tags key — append one at the end of frontmatter
        # Insert before trailing blank lines
        insert_at = len(lines)
        while insert_at > 0 and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        lines.insert(insert_at, "tags:")
        lines.insert(insert_at + 1, f"  - {new_tag}")
    else:
        tags_line = lines[tags_line_idx]

        # Flow style:  tags: [a, b, c]
        flow_match = re.match(r'^tags\s*:\s*\[(.*)]\s*$', tags_line)
        if flow_match:
            items = flow_match.group(1).strip()
            if items:
                lines[tags_line_idx] = f"tags: [{items}, {new_tag}]"
            else:
                lines[tags_line_idx] = f"tags: [{new_tag}]"
        else:
            # Block style or empty — find last list item under tags
            last_item_idx = tags_line_idx
            indent = "  "
            for j in range(tags_line_idx + 1, len(lines)):
                stripped = lines[j].strip()
                if stripped == "":
                    continue
                m = re.match(r'^(\s+)-', lines[j])
                if m:
                    last_item_idx = j
                    indent = m.group(1)
                else:
                    break  # hit a different key
            lines.insert(last_item_idx + 1, f"{indent}- {new_tag}")

    new_fm = "\n".join(lines)
    new_text = f"---{new_fm}---{body}"

    try:
        filepath.write_text(new_text, encoding="utf-8")
    except OSError as e:
        return False, f"Cannot write {filepath.name}: {e}"

    return True, f"Added tag '{new_tag}' to {filepath.name}"


def execute_tag_writeback(
    filename: str,
    tag: str,
) -> tuple[bool, str]:
    """Execute a tag write-back by finding the file and adding the tag.

    Args:
        filename: Filename (without .md) of the note to edit.
        tag: Tag string to add.

    Returns:
        (success, message) tuple.
    """
    candidates = list(VAULT_PATH.rglob(f"{filename}.md"))
    if not candidates:
        return False, f"File not found: {filename}.md"
    if len(candidates) > 1:
        return False, f"Ambiguous: multiple files named {filename}.md"

    filepath = candidates[0]
    return add_tag_to_frontmatter(filepath, tag)

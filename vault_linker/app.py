"""
Vault Linker TUI — interactive terminal interface for discovering
and ranking note connections in your Obsidian vault.
"""

import platform
import subprocess

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Footer, Header, Input, Label, ListItem, ListView, Rule, Static,
)
from textual.message import Message

import numpy as np

from vault_linker.config import (
    FOLDER_COLORS, MAX_CONNECTIONS, ACTION_HINT_THRESHOLD,
    TAG_HIGH_WEIGHT, TAG_HIGH_PEERS, TAG_MOD_WEIGHT, TAG_MOD_PEERS,
    VAULT_PATH,
)
from vault_linker.vault import Note, load_vault, load_note
from vault_linker.embeddings import compute_embeddings
from vault_linker.similarity import (
    Connection, find_connections, connection_stats,
    compute_direction_stats,
)
from vault_linker.tags import TagSuggestion, suggest_tags
from vault_linker.writeback import (
    resolve_writeback, execute_writeback, execute_tag_writeback,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def folder_color(folder: str) -> str:
    """Get the Rich color string for a folder."""
    return FOLDER_COLORS.get(folder, "white")


def score_color(score: float) -> str:
    """Color-code a similarity score."""
    if score >= 0.80:
        return "#50fa7b bold"   # bright green — strong match
    elif score >= 0.65:
        return "yellow"
    elif score >= 0.50:
        return "bright_black"
    else:
        return "dim"


def format_score(score: float) -> str:
    """Format score as percentage."""
    return f"{score:.0%}"


def direction_color(direction: str) -> str:
    """Color for directional arrows."""
    if direction == "→":
        return "#ff79c6"   # pink — outgoing reference
    elif direction == "←":
        return "#8be9fd"   # cyan — incoming reference
    else:
        return "#bd93f9"   # purple — bidirectional/related


def tag_confidence_color(weight: float, peer_count: int) -> str:
    """Color-code a tag suggestion by confidence level."""
    if weight >= TAG_HIGH_WEIGHT and peer_count >= TAG_HIGH_PEERS:
        return "#50fa7b bold"  # green — high confidence
    elif weight >= TAG_MOD_WEIGHT and peer_count >= TAG_MOD_PEERS:
        return "#f1fa8c"       # yellow — moderate
    else:
        return "#6272a4"       # dim gray — speculative


def _render_tag_line(
    suggestions: list[TagSuggestion],
    focused_idx: int = -1,
) -> str:
    """Build the Rich markup string for suggested tags, with optional highlight."""
    parts = []
    for i, s in enumerate(suggestions):
        tc = tag_confidence_color(s.weight, s.peer_count)
        tag_text = f"[{tc}]{s.tag}[/][#6272a4]({s.peer_count})[/]"
        if i == focused_idx:
            tag_text = f"[on #44475a]{tag_text}[/]"
        parts.append(tag_text)
    tags_line = "  ".join(parts)
    return f"[bold #f1fa8c]Suggested tags:[/]  {tags_line}"


def _link_status(c: Connection) -> str:
    """Build the status marker for a connection based on direction and link state.

    → (references):
        satisfied  → "✓ references"   (green)  — you correctly link to target
        above 80%  → "· add link"     (dim)    — you should link to target
    ← (referenced by):
        satisfied  → "✓ referenced"   (green)  — target correctly links to you
        above 80%  → "· needs backlink" (dim)  — target should link to you
    ↔ (related):
        both dirs  → "✓ linked"       (green)  — fully linked
        one dir    → "⚬ partial"      (yellow) — one direction missing
        above 80%  → "· add link"     (dim)    — no link yet
    """
    if c.direction == "→":
        if c.forward_linked:
            return " [#50fa7b]✓ references[/]"
        elif c.score >= ACTION_HINT_THRESHOLD:
            return " [#6272a4]· add link[/]"
    elif c.direction == "←":
        if c.backward_linked:
            return " [#50fa7b]✓ referenced[/]"
        elif c.score >= ACTION_HINT_THRESHOLD:
            return " [#6272a4]· needs backlink[/]"
    else:  # ↔
        if c.forward_linked and c.backward_linked:
            return " [#50fa7b]✓ linked[/]"
        elif c.forward_linked or c.backward_linked:
            return " [#f1fa8c]⚬ partial[/]"
        elif c.score >= ACTION_HINT_THRESHOLD:
            return " [#6272a4]· add link[/]"
    return ""


# ── Widgets ──────────────────────────────────────────────────────────────────

class NoteItem(ListItem):
    """A single note entry in the sidebar."""

    def __init__(self, note: Note) -> None:
        super().__init__()
        self.note = note

    def compose(self) -> ComposeResult:
        color = folder_color(self.note.folder)
        yield Label(
            f"[{color}]{self.note.filename}[/]",
            markup=True,
        )


class ConnectionDisplay(Static):
    """Displays a single connection result. Click to navigate."""

    def __init__(self, conn: Connection, rank: int) -> None:
        self.conn = conn
        self.rank = rank
        super().__init__()

    def compose(self) -> ComposeResult:
        c = self.conn
        color = folder_color(c.folder)
        sc = score_color(c.score)
        dc = direction_color(c.direction)

        # Direction-aware link status
        linked_marker = _link_status(c)

        tags_str = ", ".join(c.tags[:4]) if c.tags else ""

        text = (
            f"[{sc}]{format_score(c.score)}[/]  "
            f"[{dc}]{c.direction}[/]  "
            f"[{color}]{c.filename}[/]"
            f"{linked_marker}\n"
            f"      [{dc}]{c.direction_label}[/]"
        )
        if tags_str:
            text += f"  [#6272a4]│ {tags_str}[/]"

        yield Label(text, markup=True)

    def on_click(self) -> None:
        """Navigate to this connection's note when clicked."""
        self.post_message(self.Clicked(self.conn.filename))

    class Clicked(Message):
        """Posted when a connection is clicked."""
        def __init__(self, filename: str) -> None:
            self.filename = filename
            super().__init__()


HELP_TEXT = """\
[bold #f1fa8c]Vault Linker — Keybindings[/]

[bold]Navigation[/]
  [#ff79c6]j / k[/]      Navigate connections (down / up)
  [#ff79c6]h / l[/]      Navigate suggested tags (left / right)
  [#ff79c6][ / ][/]      Resize sidebar (shrink / grow)

[bold]Write[/]
  [#ff79c6]Ctrl+W[/]     Write focused connection as [[link]]
  [#ff79c6]t[/]          Write focused tag to frontmatter
  [#ff79c6]y / n[/]      Confirm / cancel a pending write
  [#ff79c6]Ctrl+B[/]     Batch write — review all suggestions

[bold]Search & Sort[/]
  [#ff79c6]/[/]          Filter notes by name, folder, or tag
  [#ff79c6]Escape[/]     Clear search
  [#ff79c6]Ctrl+S[/]     Sort: alphabetical / newest first
  [#ff79c6]Ctrl+T[/]     Show / hide already-linked notes

[bold]Other[/]
  [#ff79c6]o[/]          Open note in Obsidian
  [#ff79c6]Ctrl+R[/]     Re-embed (rebuild all embeddings)
  [#ff79c6]Ctrl+Q[/]     Quit
  [#ff79c6]F1[/]         Close this help screen
"""


class HelpScreen(ModalScreen):
    """Full-screen help overlay showing all keybindings."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    #help-box {
        width: 56;
        height: auto;
        max-height: 85%;
        padding: 1 3;
        background: #282a36;
        border: round #6272a4;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("f1", "dismiss", "Close", priority=True),
        Binding("escape", "dismiss", "Close", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Label(HELP_TEXT, markup=True, id="help-box")

    def action_dismiss(self) -> None:
        self.app.pop_screen()


class ConnectionPanel(Vertical):
    """Right panel showing connections for the selected note."""

    def update_connections(
        self,
        note: Note | None,
        connections: list[Connection],
        stats: dict,
        tag_suggestions: list[TagSuggestion] | None = None,
    ):
        """Refresh the panel with new connection data."""
        self.remove_children()

        if note is None:
            self.mount(Label("[#6272a4]Select a note to see connections[/]", markup=True))
            return

        # Header
        color = folder_color(note.folder)
        header_text = (
            f"[bold][{color}]{note.title}[/][/]\n"
            f"[#6272a4]{note.folder}/{note.filename}.md[/]\n"
        )
        if note.tags:
            header_text += f"[#6272a4]tags: {', '.join(note.tags)}[/]\n"

        if stats.get("count", 0) > 0:
            header_text += (
                f"\n[bold]Connections:[/] {stats['unlinked']} new  "
                f"[#50fa7b]{stats['linked']} linked[/]  "
                f"[#ff79c6]{stats['directional']} directional[/]  "
                f"[#6272a4]{stats['bidirectional']} mutual[/]\n"
                f"[#6272a4]scores: {format_score(stats['min_score'])} – "
                f"{format_score(stats['max_score'])}[/]"
            )

        self.mount(Label(header_text, markup=True, classes="conn-header"))

        # Tag suggestions
        if tag_suggestions:
            suggestion_text = _render_tag_line(tag_suggestions, focused_idx=-1)
            self.mount(Label(
                suggestion_text, markup=True,
                classes="conn-suggestions",
            ))

        self.mount(Rule())

        if not connections:
            self.mount(Label("[#6272a4]No connections found.[/]", markup=True))
            return

        for i, conn in enumerate(connections):
            self.mount(ConnectionDisplay(conn, i + 1))

    def show_batch_item(
        self,
        item_desc: str,
        progress: str,
        approved_count: int,
    ):
        """Display a single batch write-back suggestion for review."""
        self.remove_children()
        header = (
            f"[bold]Batch Write-Back[/]\n"
            f"[#6272a4]{progress}[/]  "
            f"[#50fa7b]{approved_count} approved[/]"
        )
        self.mount(Label(header, markup=True))
        self.mount(Rule())
        self.mount(Label(f"\n{item_desc}", markup=True))
        self.mount(Label(
            "\n[#f1fa8c]y[/] approve    "
            "[#6272a4]n[/] skip    "
            "[#ff5555]Escape[/] finish & apply",
            markup=True,
        ))


# ── Main App ─────────────────────────────────────────────────────────────────

class VaultLinkerApp(App):
    """Vault Linker — discover connections in your Obsidian vault."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 40;
        min-width: 20;
        max-width: 70;
        border-right: solid $accent;
        height: 100%;
    }

    #sidebar-header {
        height: 3;
        padding: 0 1;
        background: $surface;
    }

    #search-input {
        height: 3;
        margin: 0 1;
        display: none;
    }

    #search-input.visible {
        display: block;
    }

    #note-list {
        height: 1fr;
    }

    #main-panel {
        width: 1fr;
        height: 100%;
        padding: 1 2;
        overflow-y: auto;
    }

    .conn-header {
        margin-bottom: 0;
    }

    .conn-suggestions {
        margin-top: 0;
        margin-bottom: 0;
        padding: 0 1;
        border: round #6272a4;
        background: #282a36;
    }

    Rule {
        margin: 0;
        color: #6272a4;
    }

    ConnectionDisplay {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }

    ConnectionDisplay:hover {
        background: $surface;
    }

    ConnectionDisplay.focused {
        background: $boost;
        border-left: thick #ff79c6;
    }

    Footer {
        background: $surface;
    }
    """

    BINDINGS = [
        # ── Visible in footer ─────────────────────────────────────────
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("slash", "focus_search", "Search", key_display="/", priority=True),
        Binding("ctrl+s", "toggle_sort", "Sort", priority=True),
        Binding("ctrl+w", "writeback", "Write Link", priority=True),
        Binding("t", "write_tag", "Write Tag", priority=True),
        Binding("ctrl+b", "batch_writeback", "Batch Write", priority=True),
        Binding("o", "open_obsidian", "Obsidian", priority=True),
        Binding("f1", "toggle_help", "Help", priority=True),
        # ── Hidden (navigation) ───────────────────────────────────────
        Binding("j", "next_conn", "Next Conn", show=False),
        Binding("k", "prev_conn", "Prev Conn", show=False),
        Binding("h", "prev_tag", "Prev Tag", show=False),
        Binding("l", "next_tag", "Next Tag", show=False),
        Binding("left_square_bracket", "sidebar_shrink", "Shrink Sidebar", show=False),
        Binding("right_square_bracket", "sidebar_grow", "Grow Sidebar", show=False),
        # ── Hidden (toggles & infrequent) ─────────────────────────────
        Binding("ctrl+r", "refresh", "Re-embed", show=False),
        Binding("ctrl+t", "toggle_linked", "Hide/Show Linked", show=False),
        Binding("escape", "clear_search", "Clear Search", show=False),
        Binding("y", "confirm_writeback", "Confirm", show=False),
        Binding("n", "cancel_writeback", "Cancel", show=False),
    ]

    TITLE = "Vault Linker"
    SUB_TITLE = "Obsidian Connection Explorer"

    hide_linked: reactive[bool] = reactive(False)
    sort_mode: reactive[str] = reactive("alpha")

    def __init__(self):
        super().__init__()
        self.notes: dict[str, Note] = {}
        self.body_embeddings: dict[str, np.ndarray] = {}
        self.title_embeddings: dict[str, np.ndarray] = {}
        self.direction_stats: tuple[float, float] = (0.0, 1.0)
        self.selected_note: Note | None = None
        self._sorted_filenames: list[str] = []
        self._connections: list[Connection] = []  # current connections list
        self._focused_conn_idx: int = -1          # -1 = none focused
        self._tag_suggestions: list[TagSuggestion] = []
        self._focused_tag_idx: int = -1  # -1 = none focused
        self._pending_tag: tuple[str, str] | None = None  # (filename, tag)
        self._pending_writeback: tuple[str, str, str] | None = None  # (file, link, desc)
        self._batch_mode: bool = False
        self._batch_queue: list[tuple[str, str, str, float]] = []  # (file, link, desc, score)
        self._batch_approved: list[tuple[str, str]] = []  # (file, link)
        self._batch_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label(
                    "[bold]  Notes[/]",
                    markup=True,
                    id="sidebar-header",
                )
                yield Input(
                    placeholder="Filter notes...",
                    id="search-input",
                )
                yield ListView(id="note-list")
            yield ConnectionPanel(id="main-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Load vault and compute embeddings on startup."""
        self.notify("Loading vault...")
        self.notes = load_vault()

        self.notify(f"Found {len(self.notes)} notes. Computing embeddings...")
        self.body_embeddings, self.title_embeddings = compute_embeddings(
            self.notes,
            progress_callback=lambda msg: self.notify(msg),
        )

        self.notify("Computing direction statistics...")
        self.direction_stats = compute_direction_stats(
            self.notes, self.body_embeddings, self.title_embeddings,
        )
        mean_d, std_d = self.direction_stats
        self.notify(
            f"Direction stats: μ={mean_d:.4f} σ={std_d:.4f}",
            timeout=5,
        )

        self._populate_sidebar()
        self._update_sidebar_header(len(self.notes))
        self.notify("Ready.", timeout=3)

    def _update_sidebar_header(self, count: int) -> None:
        """Update the sidebar header with the current note count."""
        header = self.query_one("#sidebar-header", Label)
        header.update(f"[bold]  Notes ({count})[/]")

    def _populate_sidebar(self) -> None:
        """Fill the sidebar with notes grouped by folder."""
        list_view = self.query_one("#note-list", ListView)
        list_view.clear()

        by_folder: dict[str, list[Note]] = {}
        for note in self.notes.values():
            by_folder.setdefault(note.folder, []).append(note)

        if self.sort_mode == "date":
            sort_key = lambda n: n.created or "0"
            reverse = True
        else:
            sort_key = lambda n: n.filename
            reverse = False

        self._sorted_filenames = []
        for folder in sorted(by_folder.keys()):
            notes_in_folder = sorted(by_folder[folder], key=sort_key, reverse=reverse)
            for note in notes_in_folder:
                list_view.append(NoteItem(note))
                self._sorted_filenames.append(note.filename)

        self._update_sidebar_header(len(self.notes))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """When a note is selected in the sidebar, show its connections."""
        if isinstance(event.item, NoteItem):
            self.selected_note = event.item.note
            self._refresh_connections()

    def _refresh_connections(self) -> None:
        """Recompute and display connections for the selected note."""
        if not self.selected_note:
            return

        connections = find_connections(
            self.selected_note.filename,
            self.notes,
            self.body_embeddings,
            self.title_embeddings,
            self.direction_stats,
            max_results=MAX_CONNECTIONS,
            hide_linked=self.hide_linked,
        )
        self._connections = connections
        stats = connection_stats(connections)

        # Tag suggestions from peer connections
        suggestions = suggest_tags(
            self.selected_note.tags,
            connections,
            min_peers=2,
        )

        self._tag_suggestions = suggestions
        self._focused_tag_idx = -1

        panel = self.query_one("#main-panel", ConnectionPanel)
        panel.update_connections(self.selected_note, connections, stats, suggestions)
        self._focused_conn_idx = -1
        self._highlight_focused_conn()

    def _highlight_focused_conn(self) -> None:
        """Apply visual highlight to the currently focused connection."""
        try:
            panel = self.query_one("#main-panel", ConnectionPanel)
            for i, widget in enumerate(panel.query(ConnectionDisplay)):
                if i == self._focused_conn_idx and self._focused_conn_idx >= 0:
                    widget.add_class("focused")
                    widget.scroll_visible()
                else:
                    widget.remove_class("focused")
        except Exception:
            pass

    def action_next_conn(self) -> None:
        """Move focus to the next connection (j key)."""
        if self._batch_mode or not self._connections:
            return
        if self._focused_conn_idx < 0:
            self._focused_conn_idx = 0
        else:
            self._focused_conn_idx = min(
                self._focused_conn_idx + 1, len(self._connections) - 1
            )
        self._highlight_focused_conn()

    def action_prev_conn(self) -> None:
        """Move focus to the previous connection (k key)."""
        if self._batch_mode or not self._connections:
            return
        self._focused_conn_idx = max(self._focused_conn_idx - 1, 0)
        self._highlight_focused_conn()

    # ── Tag navigation ────────────────────────────────────────────────────

    def _update_tag_highlight(self) -> None:
        """Re-render the tag suggestion label with the current focus."""
        if not self._tag_suggestions:
            return
        try:
            panel = self.query_one("#main-panel", ConnectionPanel)
            label = panel.query_one(".conn-suggestions", Label)
            label.update(
                _render_tag_line(self._tag_suggestions, self._focused_tag_idx)
            )
        except Exception:
            pass

    def action_next_tag(self) -> None:
        """Move focus to the next suggested tag (l key)."""
        if self._batch_mode or not self._tag_suggestions:
            return
        if self._focused_tag_idx < 0:
            self._focused_tag_idx = 0
        else:
            self._focused_tag_idx = min(
                self._focused_tag_idx + 1, len(self._tag_suggestions) - 1
            )
        self._update_tag_highlight()

    def action_prev_tag(self) -> None:
        """Move focus to the previous suggested tag (h key)."""
        if self._batch_mode or not self._tag_suggestions:
            return
        if self._focused_tag_idx < 0:
            self._focused_tag_idx = len(self._tag_suggestions) - 1
        else:
            self._focused_tag_idx = max(self._focused_tag_idx - 1, 0)
        self._update_tag_highlight()

    def action_write_tag(self) -> None:
        """Write the currently focused suggested tag to the selected note."""
        if not self.selected_note:
            self.notify("No note selected.")
            return
        if not self._tag_suggestions:
            self.notify("No tag suggestions available.")
            return
        if self._focused_tag_idx < 0:
            self.notify("No tag focused — use h/l to select a tag first.")
            return

        tag = self._tag_suggestions[self._focused_tag_idx].tag
        desc = f"Add tag '{tag}' to {self.selected_note.filename}?"
        self._pending_tag = (self.selected_note.filename, tag)
        self.notify(f"{desc}  [y/n]", timeout=10)

    # ── Toggles ─────────────────────────────────────────────────────────────

    def action_toggle_linked(self) -> None:
        """Toggle hiding of already-linked notes."""
        self.hide_linked = not self.hide_linked
        state = "hidden" if self.hide_linked else "shown"
        self.notify(f"Already-linked notes: {state}")
        self._refresh_connections()

    def action_toggle_sort(self) -> None:
        """Toggle between alphabetical and date sort."""
        if self.sort_mode == "alpha":
            self.sort_mode = "date"
            self.notify("Sort: newest first")
        else:
            self.sort_mode = "alpha"
            self.notify("Sort: A–Z")
        self._populate_sidebar()

    def _get_sidebar_width(self) -> int:
        """Get current sidebar width, falling back to CSS default."""
        sidebar = self.query_one("#sidebar")
        w = sidebar.styles.width
        if w is not None and w.value is not None:
            return int(w.value)
        return 40  # matches CSS default

    def action_sidebar_shrink(self) -> None:
        """Shrink the sidebar by 5 columns."""
        new_width = max(20, self._get_sidebar_width() - 5)
        self.query_one("#sidebar").styles.width = new_width

    def action_sidebar_grow(self) -> None:
        """Grow the sidebar by 5 columns."""
        new_width = min(70, self._get_sidebar_width() + 5)
        self.query_one("#sidebar").styles.width = new_width

    def action_toggle_help(self) -> None:
        """Show the help screen as a modal overlay."""
        self.push_screen(HelpScreen())

    def action_focus_search(self) -> None:
        """Show and focus the search input."""
        search = self.query_one("#search-input", Input)
        search.add_class("visible")
        search.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter sidebar as user types."""
        if event.input.id == "search-input":
            self._filter_sidebar(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """On Enter, focus the note list so user can navigate results."""
        if event.input.id == "search-input":
            list_view = self.query_one("#note-list", ListView)
            list_view.focus()

    def action_clear_search(self) -> None:
        """Clear and hide the search input (or finish batch mode on Escape)."""
        if self._batch_mode:
            self._batch_finish()
            return

        search = self.query_one("#search-input", Input)
        if search.has_class("visible"):
            search.value = ""
            search.remove_class("visible")
            self._populate_sidebar()
            self.query_one("#note-list", ListView).focus()

    def _filter_sidebar(self, query: str) -> None:
        """Filter the sidebar notes by search query."""
        list_view = self.query_one("#note-list", ListView)
        list_view.clear()

        query_lower = query.lower().strip()
        if not query_lower:
            self._populate_sidebar()
            return

        if self.sort_mode == "date":
            sort_key = lambda n: n.created or "0"
            reverse = True
        else:
            sort_key = lambda n: n.filename
            reverse = False

        self._sorted_filenames = []
        by_folder: dict[str, list[Note]] = {}
        for note in self.notes.values():
            searchable = (
                f"{note.filename} {note.title} {note.folder} "
                f"{' '.join(note.tags)}"
            ).lower()
            if query_lower in searchable:
                by_folder.setdefault(note.folder, []).append(note)

        for folder in sorted(by_folder.keys()):
            notes_in_folder = sorted(by_folder[folder], key=sort_key, reverse=reverse)
            for note in notes_in_folder:
                list_view.append(NoteItem(note))
                self._sorted_filenames.append(note.filename)

        self._update_sidebar_header(len(self._sorted_filenames))

    def action_writeback(self) -> None:
        """Initiate a write-back for the currently focused connection."""
        if not self.selected_note or not self._connections:
            self.notify("No connection selected.")
            return
        if self._focused_conn_idx < 0:
            self.notify("No connection focused — use j/k to select one first.")
            return

        idx = self._focused_conn_idx
        conn = self._connections[idx]

        file_to_modify, link_to_add = resolve_writeback(
            self.selected_note.filename,
            conn.filename,
            conn.direction,
        )

        if conn.direction_satisfied:
            self.notify(f"Link already exists ({conn.direction} {conn.filename})")
            return

        desc = f"Add [[{link_to_add}]] to {file_to_modify}.md?"
        self._pending_writeback = (file_to_modify, link_to_add, desc)
        self.notify(f"{desc}  [y/n]", timeout=10)

    def action_confirm_writeback(self) -> None:
        """Confirm a pending write-back or tag write (or approve batch item)."""
        if self._batch_mode:
            self._batch_approve_current()
            return

        # Tag write-back takes priority if pending
        if self._pending_tag:
            filename, tag = self._pending_tag
            self._pending_tag = None

            success, message = execute_tag_writeback(filename, tag)
            if success:
                self.notify(f"[#50fa7b]{message}[/]", markup=True, timeout=5)
                # Reload the note so tags update
                candidates = list(VAULT_PATH.rglob(f"{filename}.md"))
                if candidates:
                    updated = load_note(candidates[0])
                    if updated:
                        self.notes[updated.filename] = updated
                self._refresh_connections()
            else:
                self.notify(f"[#ff5555]{message}[/]", markup=True, timeout=5)
            return

        if not self._pending_writeback:
            return

        file_to_modify, link_to_add, _ = self._pending_writeback
        self._pending_writeback = None

        success, message = execute_writeback(file_to_modify, link_to_add)
        if success:
            self.notify(f"[#50fa7b]{message}[/]", markup=True, timeout=5)
            # Reload the modified note so link status updates
            candidates = list(VAULT_PATH.rglob(f"{file_to_modify}.md"))
            if candidates:
                updated = load_note(candidates[0])
                if updated:
                    self.notes[updated.filename] = updated
            self._refresh_connections()
        else:
            self.notify(f"[#ff5555]{message}[/]", markup=True, timeout=5)

    def action_cancel_writeback(self) -> None:
        """Cancel a pending write-back or tag write (or skip batch item)."""
        if self._batch_mode:
            self._batch_skip_current()
            return

        if self._pending_tag:
            self._pending_tag = None
            self.notify("Tag write cancelled.", timeout=3)
            return

        if self._pending_writeback:
            self._pending_writeback = None
            self.notify("Write-back cancelled.", timeout=3)

    def action_batch_writeback(self) -> None:
        """Enter batch write-back mode: review all unlinked suggestions across the vault."""
        if self._batch_mode:
            return

        self.notify("Scanning vault for unlinked connections...")
        queue: list[tuple[str, str, str, float]] = []
        seen: set[tuple[str, str]] = set()

        for filename in self.notes:
            if filename not in self.body_embeddings:
                continue
            connections = find_connections(
                filename, self.notes,
                self.body_embeddings, self.title_embeddings,
                self.direction_stats,
                max_results=MAX_CONNECTIONS,
                hide_linked=True,
            )
            for conn in connections:
                if conn.score < ACTION_HINT_THRESHOLD:
                    continue
                if conn.direction_satisfied:
                    continue

                file_to_modify, link_to_add = resolve_writeback(
                    filename, conn.filename, conn.direction,
                )
                key = (file_to_modify, link_to_add)
                if key in seen:
                    continue
                seen.add(key)

                source_folder = self.notes.get(file_to_modify)
                fc = folder_color(source_folder.folder) if source_folder else "white"
                desc = (
                    f"[{score_color(conn.score)}]{format_score(conn.score)}[/]  "
                    f"[{direction_color(conn.direction)}]{conn.direction}[/]  "
                    f"Add [bold][[{link_to_add}]][/] to "
                    f"[{fc}]{file_to_modify}.md[/]"
                )
                queue.append((file_to_modify, link_to_add, desc, conn.score))

        if not queue:
            self.notify("No unlinked suggestions above threshold.")
            return

        queue.sort(key=lambda x: x[3], reverse=True)
        self._batch_queue = queue
        self._batch_approved = []
        self._batch_idx = 0
        self._batch_mode = True
        self.notify(f"Batch mode: {len(queue)} suggestions to review.", timeout=5)
        self._show_batch_item()

    def _show_batch_item(self) -> None:
        """Display the current batch item in the connection panel."""
        if self._batch_idx >= len(self._batch_queue):
            self._batch_finish()
            return

        _, _, desc, _ = self._batch_queue[self._batch_idx]
        progress = f"{self._batch_idx + 1} / {len(self._batch_queue)}"
        panel = self.query_one("#main-panel", ConnectionPanel)
        panel.show_batch_item(desc, progress, len(self._batch_approved))

    def _batch_approve_current(self) -> None:
        """Approve the current batch item and advance."""
        if self._batch_idx < len(self._batch_queue):
            file_to_modify, link_to_add, _, _ = self._batch_queue[self._batch_idx]
            self._batch_approved.append((file_to_modify, link_to_add))
        self._batch_idx += 1
        self._show_batch_item()

    def _batch_skip_current(self) -> None:
        """Skip the current batch item and advance."""
        self._batch_idx += 1
        self._show_batch_item()

    def _batch_finish(self) -> None:
        """Apply all approved batch writes and exit batch mode."""
        approved = list(self._batch_approved)
        self._batch_mode = False
        self._batch_queue = []
        self._batch_approved = []
        self._batch_idx = 0

        if not approved:
            self.notify("Batch finished. No links written.", timeout=5)
            self._refresh_connections()
            return

        successes = 0
        failures = 0
        for file_to_modify, link_to_add in approved:
            success, _ = execute_writeback(file_to_modify, link_to_add)
            if success:
                successes += 1
                candidates = list(VAULT_PATH.rglob(f"{file_to_modify}.md"))
                if candidates:
                    updated = load_note(candidates[0])
                    if updated:
                        self.notes[updated.filename] = updated
            else:
                failures += 1

        summary = f"Batch complete: {successes} links written"
        if failures:
            summary += f", {failures} failed"
        self.notify(summary, timeout=8)
        self._refresh_connections()

    def action_open_obsidian(self) -> None:
        """Open the currently selected note in Obsidian."""
        if not self.selected_note:
            self.notify("No note selected.")
            return

        vault_name = VAULT_PATH.name
        uri = f"obsidian://open?vault={vault_name}&file={self.selected_note.filename}"

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", uri])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", uri])
            elif system == "Windows":
                subprocess.Popen(["cmd", "/c", "start", uri], shell=True)
            self.notify(f"Opening {self.selected_note.filename} in Obsidian...", timeout=3)
        except OSError as e:
            self.notify(f"Failed to open Obsidian: {e}", timeout=5)

    def on_connection_display_clicked(self, event: ConnectionDisplay.Clicked) -> None:
        """Navigate to a connection when clicked in the right panel."""
        target_note = self.notes.get(event.filename)
        if not target_note:
            return

        self.selected_note = target_note
        self._refresh_connections()

        # Scroll sidebar to highlight the selected note
        list_view = self.query_one("#note-list", ListView)
        if event.filename in self._sorted_filenames:
            idx = self._sorted_filenames.index(event.filename)
            list_view.index = idx

    def action_refresh(self) -> None:
        """Force re-embed all notes (clears cache)."""
        self.notify("Re-embedding all notes...")

        self.notes = load_vault()

        from vault_linker.config import CACHE_FILE, TITLE_CACHE_FILE
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        if TITLE_CACHE_FILE.exists():
            TITLE_CACHE_FILE.unlink()

        self.body_embeddings, self.title_embeddings = compute_embeddings(
            self.notes,
            progress_callback=lambda msg: self.notify(msg),
        )
        self.direction_stats = compute_direction_stats(
            self.notes, self.body_embeddings, self.title_embeddings,
        )
        self._populate_sidebar()
        self._refresh_connections()
        self.notify("Re-embed complete.", timeout=3)


def run():
    """Entry point."""
    app = VaultLinkerApp()
    app.run()

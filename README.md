# Vault Linker

Terminal UI for discovering, ranking, and managing note connections in your Obsidian vault using local semantic embeddings.

Computes dual embeddings per note (body content + title/tags identity), uses asymmetric body-to-title cosine similarity to infer link direction, and surfaces tag suggestions from peer propagation.

## Setup

```bash
cd vault-automation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your vault path:

```bash
export VAULT_PATH=~/path/to/your/ObsidianVault
```

Requires a running Ollama instance with an embedding model:

```bash
ollama pull nomic-embed-text
python run.py
```

## Configuration

All settings live in `vault_linker/config.py`:

**Paths and provider:**
- `VAULT_PATH` — path to your Obsidian vault
- `EMBEDDING_PROVIDER` — `"openai"` or `"ollama"` (default: `"ollama"`)
- `OLLAMA_MODEL` — embedding model name (default: `"nomic-embed-text"`)
- `EXCLUDE_DIRS` — folders to skip (Obsidian internals pre-set; add your own via `VAULT_EXCLUDE_DIRS` env var)

**Thresholds:**
- `ACTION_HINT_THRESHOLD` (0.75) — minimum score to show "add link" hints
- `DIRECTION_Z_THRESHOLD` (1.5) — z-score cutoff for directional inference
- `MAX_CONNECTIONS` (20) — results shown per note

**Tag suggestion confidence:**
- `TAG_HIGH_WEIGHT` / `TAG_HIGH_PEERS` — green (high confidence)
- `TAG_MOD_WEIGHT` / `TAG_MOD_PEERS` — yellow (moderate)
- Below moderate thresholds — dim gray (speculative)

## Keybindings

| Key | Action |
|-----|--------|
| `↑`/`↓` | Navigate notes in sidebar |
| `Enter` | Select note, show connections |
| `/` | Search/filter notes |
| `Escape` | Clear search |
| `Ctrl+T` | Hide/show already-linked notes |
| `Ctrl+S` | Toggle sort: alphabetical / newest first |
| `j` / `k` | Navigate connections in right panel |
| `Ctrl+W` | Write link — add `[[link]]` to See Also (with confirmation) |
| `Ctrl+B` | Batch write-back — review all unlinked suggestions vault-wide |
| `Y` / `N` | Confirm / cancel (or approve / skip in batch mode) |
| `O` | Open focused connection in Obsidian |
| `Ctrl+R` | Re-embed all notes (clears cache) |
| `Ctrl+Q` | Quit |

Click any connection in the right panel to navigate to that note. Use `j`/`k` to move focus between connections — the focused connection (highlighted with a pink left border) is the target for `Ctrl+W` and `O`.

## How It Works

### Dual Embedding System

Each note produces two embeddings:

- **Body embedding:** note content with frontmatter stripped. Captures what the note *discusses*.
- **Title embedding:** note title + tags. Captures what the note *is* (its semantic identity).

Body-to-body cosine similarity ranks overall relatedness. Body-to-title asymmetry infers link direction.

### Directional Inference

For notes A and B:
- `forward` = cosine(A_body, B_title) — how much A discusses B's concept
- `backward` = cosine(B_body, A_title) — how much B discusses A's concept
- `delta` = forward - backward

The delta is converted to a z-score against the vault-wide distribution. Significant positive z means A references B (`→`), significant negative means B references A (`←`), otherwise bidirectional (`↔`).

### Link Status

Connection display shows direction-aware link status:
- `✓ references` (green) — outgoing link exists and is correct
- `✓ referenced` (green) — incoming backlink exists
- `✓ linked` (green) — bidirectional connection fully linked
- `⚬ partial` (yellow) — one direction of a bidirectional link missing
- `· add link` / `· needs backlink` (dim) — suggested action for high-scoring unlinked pairs

### Tag Suggestions

Tags are suggested via peer propagation: for each connection's tags that the source note lacks, the connection's similarity score is accumulated as weight. Tags appearing across many high-scoring connections rank highest. Confidence is color-coded:
- **Green** — high weight, 3+ peers share this tag
- **Yellow** — moderate weight, 2+ peers
- **Dim gray** — speculative, single peer or low weight

### Write-Back

`Ctrl+W` initiates a write-back for the top connection. The TUI determines which file to modify based on direction:
- `→` (references): adds `[[target]]` to the source note's See Also
- `←` (referenced by): adds `[[source]]` to the target note's See Also
- `↔` (related): adds `[[target]]` to the source note's See Also

A confirmation prompt (`y`/`n`) prevents accidental writes. The modified note is reloaded immediately so link status updates in place.

### Batch Write-Back

`Ctrl+B` enters batch mode: the TUI scans every note in the vault for unlinked connections above the `ACTION_HINT_THRESHOLD`, deduplicates them, and presents them one at a time sorted by score. Press `y` to approve a suggestion, `n` to skip, or `Escape` to finish early. When the queue is exhausted (or you press Escape), all approved links are written at once and modified notes are reloaded. This is useful when onboarding a batch of new notes that need linking across the vault.

## Score Color Coding

| Score | Color | Meaning |
|-------|-------|---------|
| 80%+ | Green (bold) | Strong match |
| 65–80% | Yellow | Moderate relatedness |
| 50–65% | Gray | Weak signal |
| <50% | Dim | Likely unrelated |

## Direction Arrow Colors

| Arrow | Color | Meaning |
|-------|-------|---------|
| `→` | Pink | Source references target |
| `←` | Cyan | Target references source |
| `↔` | Purple | Bidirectional / mutual |

## Folder Colors

Folders are color-coded to match Obsidian graph view groups. Edit `FOLDER_COLORS` in `config.py` to match your vault's top-level folders.

| Folder | Color | Hex |
|--------|-------|-----|
| Topic A | Bright green | `#50fa7b` |
| Topic B | Bright cyan | `#8be9fd` |
| Topic C | Lighter cyan | `#6bc5f0` |
| Topic D | Bright pink | `#ff79c6` |
| Topic E | Bright yellow | `#f1fa8c` |
| Topic F | Bright red | `#ff5555` |
| Topic G | Muted slate | `#6272a4` |

### Additional Colors

| Color | Hex |
|-------|-----|
| Amber-orange | `#ff9f1c` |
| Warm red | `#e63946` |
| Sandy amber | `#f4a261` |
| Soft pink | `#e091c9` |
| Cool slate | `#6b7d8d` |
| Coral | `#ff6b6b` |
| Powder blue | `#a8d8ea` |
| Mint green | `#c3f0ca` |

Uncomment the relevant line in `config.py` when adding a new folder.

## Embedding Model

Default: `nomic-embed-text` (768 dims, 137M params) — purpose-built for English retrieval, performs well on MTEB benchmarks, and is widely available via Ollama.

To use a different model (e.g. `qwen3-embedding:4b` for more representational capacity on higher-end hardware):

```bash
export OLLAMA_MODEL=qwen3-embedding:4b
ollama pull qwen3-embedding:4b
```

> **Note:** `nomic-embed-text` has a known instability in some Ollama versions where its internal worker crashes under sustained load (EOF errors). If you hit this, switching to `qwen3-embedding:4b` resolves it.

### Switching models

```bash
ollama pull nomic-embed-text  # or any other model
```

1. Set `OLLAMA_MODEL` env var (or edit `config.py`)
2. Delete the `vault_linker/cache/` directory (or use `vault --reindex`)
3. Restart — all notes will re-embed with the new model

### Batch embedding

The Ollama integration uses the `/api/embed` endpoint with batch input, sending up to 10 texts per request. This is significantly faster than the legacy one-at-a-time approach. If your Ollama version is older and does not support `/api/embed`, the system falls back to single-request mode automatically.

Notes longer than ~24,000 characters are truncated before embedding to stay within the model's 8192-token context window. This is conservative (most notes are well under this limit) and preserves the opening sections where the strongest semantic signal lives. If Ollama returns an error, the actual error body is captured and printed to help diagnose the issue.

## Roadmap

If the TUI isn't surfacing many "add link" suggestions, three factors are likely compounding:

1. **Your vault is already well-linked.** If you spent time manually building backlinks and See Also sections, the tool is looking for connections you missed — and there may not be many left.
2. **Domain clustering.** Notes within the same subject area share heavy vocabulary overlap, producing a flat similarity landscape where everything is moderately similar to everything else in that domain. The tool is most valuable for **cross-domain** connections — notes from different subject areas that share an underlying concept.
3. **Embedding model capacity.** A smaller model may not distinguish subtle conceptual relationships from surface vocabulary overlap. Upgrading to a larger model (see "When to upgrade" above) improves discrimination within dense domains.

## Roadmap

Implemented:
- Core TUI with note list, connection panel, similarity scores, directional arrows
- Tag suggestion system via peer-based propagation with confidence coloring
- Write-back: `Ctrl+W` to add `[[links]]` to See Also with confirmation prompt
- Batch write-back: `Ctrl+B` to review and approve all unlinked suggestions vault-wide
- Connection panel navigation: `j`/`k` to move focus, visual highlight with pink border
- Open in Obsidian: `O` to jump to a connection in the desktop app
- Embedding cache with hash-based invalidation
- Search/filter across notes
- Hide-linked toggle and sort modes
- Batch embedding via Ollama `/api/embed` endpoint

Planned:
- **Configurable thresholds via CLI flags or `.vault-linker.toml`** — tune `ACTION_HINT_THRESHOLD`, `DIRECTION_Z_THRESHOLD`, tag confidence levels without editing source
- **Tag confidence color gradient refinement** — the confidence colors work (green/yellow/gray via `tag_confidence_color`), but the visual presentation in the TUI could be more prominent
- **Export mode** — `--report` CLI flag to dump top N unlinked connections to stdout without launching the full TUI

## Project Structure

```
vault_linker/
├── app.py          # Textual TUI (interface, keybindings, panels)
├── config.py       # All configurable settings
├── embeddings.py   # Dual embedding engine with caching
├── similarity.py   # Cosine similarity, direction inference, connection ranking
├── tags.py         # Peer-based tag propagation engine
├── vault.py        # Vault parser (frontmatter, wiki-links, See Also)
├── writeback.py    # Write-back engine (See Also link insertion)
└── cache/          # Embedding cache (auto-generated)
```


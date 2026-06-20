"""
Similarity engine: cosine similarity, asymmetric directionality,
and statistical threshold for direction inference.

Directionality logic:
  For notes A and B, compute:
    forward  = sim(A_body, B_title)  — how much A discusses B's concept
    backward = sim(B_body, A_title)  — how much B discusses A's concept
    delta    = forward - backward

  If delta is significantly positive (z > threshold):
    A references B → link goes A → B
  If delta is significantly negative (z < -threshold):
    B references A → link goes B → A (shown as ← from A's perspective)
  Otherwise:
    bidirectional ↔

  The threshold is determined statistically from the distribution of
  all pairwise deltas in the vault — self-calibrating as the vault grows.
"""

from dataclasses import dataclass

import numpy as np

from vault_linker.config import DIRECTION_Z_THRESHOLD, MAX_CONNECTIONS
from vault_linker.vault import Note, extract_see_also_links


@dataclass
class Connection:
    """A ranked connection between two notes."""
    filename: str
    folder: str
    title: str
    score: float          # body-to-body cosine similarity (overall relatedness)
    direction: str        # "→" (outgoing), "←" (incoming), "↔" (bidirectional)
    direction_label: str  # human-readable explanation
    direction_delta: float  # raw asymmetric delta (for debugging/tuning)
    forward_linked: bool  # source's See Also mentions target
    backward_linked: bool  # target's See Also mentions source
    tags: list[str]

    @property
    def already_linked(self) -> bool:
        """Whether any link exists between these notes (either direction)."""
        return self.forward_linked or self.backward_linked

    @property
    def direction_satisfied(self) -> bool:
        """Whether the link that *should* exist for this direction does exist.

        → (references):     source should link to target → check forward_linked
        ← (referenced by):  target should link to source → check backward_linked
        ↔ (related):        either direction counts
        """
        if self.direction == "→":
            return self.forward_linked
        elif self.direction == "←":
            return self.backward_linked
        else:
            return self.forward_linked or self.backward_linked


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute_direction_stats(
    notes: dict[str, Note],
    body_embeddings: dict[str, np.ndarray],
    title_embeddings: dict[str, np.ndarray],
) -> tuple[float, float]:
    """
    Compute the mean and std of asymmetric deltas across all note pairs.

    This gives us the distribution under "no directional preference"
    so we can flag significant deviations using z-scores.

    Returns (mean_delta, std_delta).
    """
    filenames = list(body_embeddings.keys())
    n = len(filenames)

    if n < 3:
        return 0.0, 1.0  # not enough data, use unit std to avoid div/0

    # Sample pairs for efficiency — all pairs if small vault,
    # random sample if large
    deltas = []

    if n <= 200:
        # Compute all pairs (manageable)
        for i in range(n):
            fn_a = filenames[i]
            if fn_a not in title_embeddings:
                continue
            for j in range(i + 1, n):
                fn_b = filenames[j]
                if fn_b not in title_embeddings:
                    continue

                forward = cosine_similarity(
                    body_embeddings[fn_a], title_embeddings[fn_b]
                )
                backward = cosine_similarity(
                    body_embeddings[fn_b], title_embeddings[fn_a]
                )
                deltas.append(forward - backward)
    else:
        # Random sample of ~5000 pairs
        rng = np.random.default_rng(42)
        for _ in range(5000):
            i, j = rng.choice(n, size=2, replace=False)
            fn_a, fn_b = filenames[i], filenames[j]
            if fn_a not in title_embeddings or fn_b not in title_embeddings:
                continue

            forward = cosine_similarity(
                body_embeddings[fn_a], title_embeddings[fn_b]
            )
            backward = cosine_similarity(
                body_embeddings[fn_b], title_embeddings[fn_a]
            )
            deltas.append(forward - backward)

    if not deltas:
        return 0.0, 1.0

    arr = np.array(deltas)
    return float(np.mean(arr)), float(np.std(arr))


def infer_direction(
    source_body: np.ndarray,
    source_title: np.ndarray,
    target_body: np.ndarray,
    target_title: np.ndarray,
    mean_delta: float,
    std_delta: float,
    z_threshold: float = DIRECTION_Z_THRESHOLD,
) -> tuple[str, str, float]:
    """
    Infer link direction using asymmetric body↔title similarity.

    Returns (arrow, label, raw_delta):
        "→"  = source references target (source body discusses target's concept)
        "←"  = target references source (target body discusses source's concept)
        "↔"  = bidirectional (no significant asymmetry)
    """
    # How much does source's body discuss target's identity?
    forward = cosine_similarity(source_body, target_title)
    # How much does target's body discuss source's identity?
    backward = cosine_similarity(target_body, source_title)

    delta = forward - backward

    # Z-score relative to vault-wide delta distribution
    if std_delta > 0:
        z = (delta - mean_delta) / std_delta
    else:
        z = 0.0

    if z > z_threshold:
        return "→", "references", delta
    elif z < -z_threshold:
        return "←", "referenced by", delta
    else:
        return "↔", "related", delta


def find_connections(
    source_filename: str,
    notes: dict[str, Note],
    body_embeddings: dict[str, np.ndarray],
    title_embeddings: dict[str, np.ndarray],
    direction_stats: tuple[float, float],
    max_results: int = MAX_CONNECTIONS,
    hide_linked: bool = False,
) -> list[Connection]:
    """
    Find and rank connections for a given note.

    Uses body-to-body similarity for ranking (overall relatedness)
    and body-to-title asymmetry for directionality.
    """
    if source_filename not in body_embeddings or source_filename not in notes:
        return []

    source_note = notes[source_filename]
    source_body = body_embeddings[source_filename]
    source_title = title_embeddings.get(source_filename)

    if source_title is None:
        return []

    mean_delta, std_delta = direction_stats
    source_see_also = set(extract_see_also_links(source_note.content))

    connections: list[Connection] = []

    for target_filename, target_body in body_embeddings.items():
        if target_filename == source_filename:
            continue

        target_note = notes.get(target_filename)
        target_title = title_embeddings.get(target_filename)
        if not target_note or target_title is None:
            continue

        # Overall relatedness score (symmetric body-to-body)
        score = cosine_similarity(source_body, target_body)

        # Check links in both directions
        forward_linked = (
            target_filename in source_see_also
            or target_note.title in source_see_also
        )
        target_see_also = set(extract_see_also_links(target_note.content))
        backward_linked = (
            source_filename in target_see_also
            or source_note.title in target_see_also
        )

        if hide_linked and (forward_linked or backward_linked):
            continue

        # Directional inference (asymmetric body↔title)
        arrow, label, delta = infer_direction(
            source_body, source_title,
            target_body, target_title,
            mean_delta, std_delta,
        )

        connections.append(Connection(
            filename=target_filename,
            folder=target_note.folder,
            title=target_note.title,
            score=score,
            direction=arrow,
            direction_label=label,
            direction_delta=delta,
            forward_linked=forward_linked,
            backward_linked=backward_linked,
            tags=target_note.tags,
        ))

    connections.sort(key=lambda c: c.score, reverse=True)
    return connections[:max_results]


def connection_stats(connections: list[Connection]) -> dict:
    """Summary statistics for a set of connections."""
    if not connections:
        return {"count": 0}

    scores = [c.score for c in connections]
    linked = sum(1 for c in connections if c.already_linked)
    unlinked = len(connections) - linked
    directional = sum(1 for c in connections if c.direction != "↔")

    return {
        "count": len(connections),
        "linked": linked,
        "unlinked": unlinked,
        "directional": directional,
        "bidirectional": len(connections) - directional,
        "max_score": max(scores),
        "min_score": min(scores),
        "mean_score": sum(scores) / len(scores),
    }

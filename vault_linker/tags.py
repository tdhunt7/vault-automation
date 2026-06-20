"""
Tag suggestion engine: peer-based tag propagation.

Suggests tags for a note by analyzing the tags of its nearest
neighbors (connections), weighted by similarity score. Tags that
appear frequently on high-scoring connections get the strongest
recommendation.

Scoring:
  For each connection's tag (that the source note doesn't have):
    weight += connection.score

  Final suggestions are ranked by total weight, which naturally
  favors tags that appear on many similar notes and/or on the
  most similar notes.

Filtering:
  - Tags the source note already has are excluded
  - A minimum weight threshold filters noise from distant connections
"""

from dataclasses import dataclass
from collections import defaultdict

from vault_linker.similarity import Connection


@dataclass
class TagSuggestion:
    """A suggested tag with supporting evidence."""
    tag: str
    weight: float        # sum of similarity scores from connections with this tag
    peer_count: int      # how many connections have this tag
    top_peer: str        # filename of highest-scoring connection with this tag
    top_peer_score: float


def suggest_tags(
    source_tags: list[str],
    connections: list[Connection],
    max_suggestions: int = 8,
    min_weight: float = 0.5,
    min_peers: int = 1,
) -> list[TagSuggestion]:
    """
    Suggest tags for a note based on its connections.

    Args:
        source_tags: Tags the note already has (will be excluded).
        connections: Ranked connections for this note.
        max_suggestions: Maximum number of suggestions to return.
        min_weight: Minimum aggregate weight to include a suggestion.
                    Filters out tags from a single weak connection.
        min_peers: Minimum number of connections that must share the tag.

    Returns:
        Ranked list of TagSuggestion objects.
    """
    existing = set(t.lower() for t in source_tags)
    canonical: dict[str, str] = {}

    # Accumulate evidence per tag
    tag_weights: dict[str, float] = defaultdict(float)
    tag_counts: dict[str, int] = defaultdict(int)
    tag_best_peer: dict[str, tuple[str, float]] = {}

    for conn in connections:
        for tag in conn.tags:
            tag_key = tag.lower()
            if tag_key in existing:
                continue

            tag_weights[tag_key] += conn.score
            tag_counts[tag_key] += 1

            if tag_key not in tag_best_peer or conn.score > tag_best_peer[tag_key][1]:
                tag_best_peer[tag_key] = (conn.filename, conn.score)

            if tag_key not in canonical:
                canonical[tag_key] = tag

    # Filter and rank
    suggestions = []
    for tag_key, weight in tag_weights.items():
        if weight < min_weight:
            continue
        if tag_counts[tag_key] < min_peers:
            continue

        best_fn, best_score = tag_best_peer[tag_key]
        suggestions.append(TagSuggestion(
            tag=canonical.get(tag_key, tag_key),
            weight=weight,
            peer_count=tag_counts[tag_key],
            top_peer=best_fn,
            top_peer_score=best_score,
        ))

    suggestions.sort(key=lambda s: s.weight, reverse=True)
    return suggestions[:max_suggestions]

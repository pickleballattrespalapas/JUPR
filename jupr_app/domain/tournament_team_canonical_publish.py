from __future__ import annotations

from typing import Any


NOT_RATED = "NOT_RATED"
NOT_READY = "NOT_READY"
READY_TO_PUBLISH = "READY_TO_PUBLISH"
PUBLISHED = "PUBLISHED"
RECONCILE_REQUIRED = "RECONCILE_REQUIRED"


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _side_players(
    row: dict[str, Any],
    *,
    first_key: str,
    second_key: str,
    expected_size: int,
) -> tuple[int, ...] | None:
    first = _integer(row.get(first_key))
    second = _integer(row.get(second_key))
    if first is None:
        return None
    if expected_size == 1:
        return (first,) if second is None else None
    if expected_size != 2 or second is None or first == second:
        return None
    return tuple(sorted((first, second)))


def _child_side_players(
    child: dict[str, Any],
    *,
    key: str,
    expected_size: int,
) -> tuple[int, ...] | None:
    raw = child.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != expected_size:
        return None
    players = tuple(_integer(value) for value in raw)
    if any(value is None for value in players):
        return None
    normalized = tuple(sorted(int(value) for value in players if value is not None))
    if len(set(normalized)) != expected_size:
        return None
    return normalized


def _source_shape_is_valid(
    child: dict[str, Any],
    tournament_game: dict[str, Any] | None,
) -> bool:
    if not tournament_game:
        return False
    child_id = str(child.get("id") or "")
    game_id = str(child.get("tournament_game_id") or "")
    if (
        not child_id
        or not game_id
        or str(tournament_game.get("id") or "") != game_id
        or str(tournament_game.get("team_match_game_id") or "") != child_id
        or _truthy(tournament_game.get("parent_result_only"))
    ):
        return False
    match_format = str(child.get("match_format") or "").strip().upper()
    expected_size = (
        1 if match_format == "SINGLES" else 2 if match_format == "DOUBLES" else 0
    )
    child_a = _child_side_players(
        child,
        key="team_a_player_ids",
        expected_size=expected_size,
    )
    child_b = _child_side_players(
        child,
        key="team_b_player_ids",
        expected_size=expected_size,
    )
    if (
        not expected_size
        or child_a is None
        or child_b is None
        or not set(child_a).isdisjoint(child_b)
    ):
        return False
    score_a = _integer(child.get("score_a"))
    score_b = _integer(child.get("score_b"))
    return (
        score_a is not None
        and score_b is not None
        and score_a != score_b
        and _integer(tournament_game.get("score_a")) == score_a
        and _integer(tournament_game.get("score_b")) == score_b
    )


def _canonical_matches_child(
    child: dict[str, Any],
    canonical_match: dict[str, Any],
) -> bool:
    game_id = str(child.get("tournament_game_id") or "")
    if (
        not game_id
        or str(canonical_match.get("tournament_game_id") or "") != game_id
        or canonical_match.get("deleted_at")
        or _truthy(canonical_match.get("excluded_from_ratings"))
    ):
        return False

    match_format = str(child.get("match_format") or "").strip().upper()
    expected_size = (
        1 if match_format == "SINGLES" else 2 if match_format == "DOUBLES" else 0
    )
    if (
        not expected_size
        or str(canonical_match.get("match_format") or "").strip().upper()
        != match_format
    ):
        return False

    child_a = _child_side_players(
        child,
        key="team_a_player_ids",
        expected_size=expected_size,
    )
    child_b = _child_side_players(
        child,
        key="team_b_player_ids",
        expected_size=expected_size,
    )
    official_a = _side_players(
        canonical_match,
        first_key="t1_p1",
        second_key="t1_p2",
        expected_size=expected_size,
    )
    official_b = _side_players(
        canonical_match,
        first_key="t2_p1",
        second_key="t2_p2",
        expected_size=expected_size,
    )
    return (
        child_a is not None
        and child_b is not None
        and set(child_a).isdisjoint(child_b)
        and child_a == official_a
        and child_b == official_b
        and _integer(canonical_match.get("score_t1"))
        == _integer(child.get("score_a"))
        and _integer(canonical_match.get("score_t2"))
        == _integer(child.get("score_b"))
    )


def classify_team_child_publish_state(
    *,
    child: dict[str, Any],
    tournament_game: dict[str, Any] | None,
    canonical_matches: list[dict[str, Any]],
) -> str:
    """Classify one rated child against its immutable official-match history.

    Any canonical history is authoritative. It may never fall back to ready:
    deleted, excluded, duplicate, side-swapped, malformed, or corrected history
    must go through reconciliation.
    """

    has_canonical_history = bool(canonical_matches)
    if not _truthy(child.get("counts_for_rating")):
        return RECONCILE_REQUIRED if has_canonical_history else NOT_RATED
    if str(child.get("status") or "").strip().upper() != "FINAL":
        return RECONCILE_REQUIRED if has_canonical_history else NOT_READY
    if not _source_shape_is_valid(child, tournament_game):
        return RECONCILE_REQUIRED
    if not canonical_matches:
        return READY_TO_PUBLISH
    if len(canonical_matches) != 1:
        return RECONCILE_REQUIRED
    if _canonical_matches_child(child, canonical_matches[0]):
        return PUBLISHED
    return RECONCILE_REQUIRED

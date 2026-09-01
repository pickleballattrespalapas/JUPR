from __future__ import annotations

from collections import Counter
from typing import Any

from jupr_app.domain.player_activity import coerce_utc_datetime
from jupr_app.domain.tournament_admin_operations import (
    stable_tournament_admin_fingerprint,
)
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.domain.tournaments import compute_round_robin_standings
from jupr_app.services.admin_tournament_ops_service import build_admin_tournament_ops_runtime_status
from jupr_app.services.admin_tournament_podium_review_service import (
    build_admin_tournament_podium_review_fingerprint,
    find_current_admin_tournament_podium_review,
)
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
)


ACTIVE_OPERATION_STATUSES = {"intent", "mutated", "recovery_required"}
UNCERTAIN_OPERATION_STATUSES = {"mutated", "recovery_required"}
UNSETTLED_MATCH_EXCLUSION_STATUSES = {
    "pending_replay",
    "pending_badge_reconcile",
    "recovery_required",
}
UNSETTLED_REPLAY_JOB_STATUSES = {"pending", "running"}
PROTECTED_DRAW_KINDS = {"TEAM_PARENT", "TEAM_RATING_CHILD"}
OFFICIAL_PUBLISH_ACTIONS = {
    "ops_official_publish",
    "tournament_live_official_publish",
}
OFFICIAL_MATCH_CLASSIFICATION_FIELDS = ("match_type", "league", "date", "week_tag")
INACTIVE_DRAW_STATUSES = {
    "archived",
    "cancelled",
    "canceled",
    "deleted",
    "disabled",
    "inactive",
    "void",
    "voided",
}
TOURNAMENT_LIFECYCLE_CONTRACT = "jupr:tournament-lifecycle:v1"
TOURNAMENT_LIFECYCLE_AUTHORITY = "python_fastapi"


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _enabled(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled", "active"}


def _read_rows(
    supabase: Any,
    table_name: str,
    *,
    filters: tuple[tuple[str, Any], ...],
    in_filters: tuple[tuple[str, tuple[Any, ...]], ...] = (),
    limit: int = 5000,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    try:
        safe_limit = max(1, int(limit))
        page_size = min(500, safe_limit)
        rows: list[dict[str, Any]] = []
        offset = 0
        seen_page_fingerprints: set[str] = set()
        seen_row_ids: set[str] = set()

        while True:
            query = supabase.table(table_name).select("*")
            for key, value in filters:
                query = query.eq(str(key), value)
            for key, values in in_filters:
                query = query.in_(str(key), list(values))
            if hasattr(query, "order"):
                query = query.order("id", desc=False)
            supports_range = hasattr(query, "range")
            if supports_range:
                query = query.range(offset, offset + page_size - 1)
            else:
                query = query.limit(safe_limit)

            data = getattr(query.execute(), "data", None)
            if not isinstance(data, list) or any(
                not isinstance(row, dict) for row in data
            ):
                raise RuntimeError(f"{table_name} lifecycle evidence is unavailable")
            page = [dict(row) for row in data]

            if not supports_range:
                if len(page) >= safe_limit:
                    raise RuntimeError(
                        f"{table_name} exceeded the safe lifecycle read bound"
                    )
                return page, True, None
            if not page:
                return rows, True, None

            page_fingerprint = stable_tournament_admin_fingerprint(page)
            if page_fingerprint in seen_page_fingerprints:
                raise RuntimeError(f"{table_name} lifecycle pagination repeated a page")
            seen_page_fingerprints.add(page_fingerprint)
            page_row_ids = [str(row.get("id") or "") for row in page]
            if any(not row_id for row_id in page_row_ids) or any(
                row_id in seen_row_ids for row_id in page_row_ids
            ):
                raise RuntimeError(
                    f"{table_name} lifecycle pagination returned unstable row identity"
                )
            seen_row_ids.update(page_row_ids)
            rows.extend(page)
            if len(rows) >= safe_limit:
                raise RuntimeError(
                    f"{table_name} exceeded the safe lifecycle read bound"
                )
            # Advance by the rows actually returned. PostgREST may cap a
            # requested range below page_size; stopping on a short page or
            # advancing by the requested size would silently skip evidence.
            offset += len(page)
    except Exception:
        return [], False, f"{table_name} evidence is unavailable."


def _is_primary_draw(draw: dict[str, Any]) -> bool:
    return (
        not bool(draw.get("hidden_from_primary_ops"))
        and str(draw.get("draw_kind") or "STANDARD").upper() not in PROTECTED_DRAW_KINDS
    )


def _is_active_primary_draw(draw: dict[str, Any]) -> bool:
    return _is_primary_draw(draw) and str(
        draw.get("status") or "draft"
    ).strip().lower() not in INACTIVE_DRAW_STATUSES


def _is_finalized_game(game: dict[str, Any], *, team_ids: set[str]) -> bool:
    score_a = _safe_int(game.get("score_a"))
    score_b = _safe_int(game.get("score_b"))
    team_a = str(game.get("team_a_id") or "")
    team_b = str(game.get("team_b_id") or "")
    winner = str(game.get("winner_team_id") or "")
    loser = str(game.get("loser_team_id") or "")
    if (
        score_a is None
        or score_b is None
        or score_a < 0
        or score_b < 0
        or score_a == score_b
        or not game.get("finalized_at")
        or not team_a
        or not team_b
        or team_a == team_b
        or team_a not in team_ids
        or team_b not in team_ids
    ):
        return False
    expected_winner = team_a if score_a > score_b else team_b
    expected_loser = team_b if expected_winner == team_a else team_a
    return winner == expected_winner and loser == expected_loser


def _is_rating_publish_eligible(game: dict[str, Any]) -> bool:
    """Exclude non-played outcomes from official rating publication."""

    result_type = str(game.get("result_type") or "PLAYED").strip().upper()
    return (
        result_type == "PLAYED"
        and game.get("rating_publish_eligible") is not False
        and not _enabled(game.get("parent_result_only"), default=False)
    )


def _tournament_game_scoring_format(game: dict[str, Any]) -> str:
    score_review = game.get("score_review_json")
    reviewed_format = (
        score_review.get("scoring_format")
        if isinstance(score_review, dict)
        else None
    )
    return str(reviewed_format or game.get("scoring_format") or "").strip().upper()


def _competition_game_sort_key(game: dict[str, Any]) -> tuple[Any, ...]:
    stage = str(game.get("stage") or "").strip().upper()
    stage_rank = {"ROUND_ROBIN": 0, "PLAYOFF": 1}.get(stage, 2)
    return (
        stage_rank,
        int(_safe_int(game.get("rr_round_number")) or 0),
        int(_safe_int(game.get("rr_slot_number")) or 0),
        str(game.get("playoff_game_code") or ""),
        str(game.get("created_at") or ""),
        str(game.get("id") or ""),
    )


def _series_validation_error(
    code: str,
    message: str,
    *,
    parent_game_id: str = "",
    child_game_id: str = "",
) -> dict[str, str]:
    return {
        "code": str(code),
        "message": str(message),
        "parent_game_id": str(parent_game_id),
        "child_game_id": str(child_game_id),
    }


def build_tournament_rating_game_plan(
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate competition results from canonical rating-game leaves.

    A finalized best-two-of-three matchup remains one aggregate competition
    row for standings and bracket progression.  Its finalized ``SERIES_GAME``
    children are the only rating evidence.  A retirement may preserve one or
    two completed children while the parent becomes a synthetic tournament
    loss; those children remain rated but are never marked as series-clinching.
    This validator intentionally fails closed: incomplete, orphaned, or
    internally inconsistent series children are reported and never returned
    as publishable rating games.
    """

    rows = [dict(row) for row in games]
    child_rows = [
        row
        for row in rows
        if str(row.get("series_parent_game_id") or "")
        or str(row.get("stage") or "").strip().upper() == "SERIES_GAME"
    ]
    competition_games = [row for row in rows if row not in child_rows]
    competition_game_ids = {
        str(row.get("id") or "")
        for row in competition_games
        if str(row.get("id") or "")
    }
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []
    preinvalid_parent_ids: set[str] = set()

    for child in child_rows:
        child_id = str(child.get("id") or "")
        parent_id = str(child.get("series_parent_game_id") or "")
        if not child_id:
            if parent_id:
                preinvalid_parent_ids.add(parent_id)
            errors.append(
                _series_validation_error(
                    "SERIES_GAME_ID_MISSING",
                    "A series game is missing its canonical tournament game id.",
                    parent_game_id=parent_id,
                )
            )
        if not parent_id:
            errors.append(
                _series_validation_error(
                    "SERIES_GAME_PARENT_MISSING",
                    f"Series game {child_id or '[missing id]'} has no parent matchup.",
                    child_game_id=child_id,
                )
            )
            continue
        children_by_parent.setdefault(parent_id, []).append(child)

    rating_entries: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    invalid_parent_ids: set[str] = set()

    for parent in competition_games:
        parent_id = str(parent.get("id") or "")
        scoring_format = _tournament_game_scoring_format(parent)
        result_type = str(parent.get("result_type") or "PLAYED").strip().upper()
        finalized = parent.get("finalized_at") not in (None, "")
        parent_result_only = _enabled(
            parent.get("parent_result_only"), default=False
        )
        is_finalized_best_of_three = (
            scoring_format == "BEST_2_OF_3"
            and result_type == "PLAYED"
            and finalized
        )
        series_children = children_by_parent.get(parent_id, [])
        parent_review = (
            parent.get("score_review_json")
            if isinstance(parent.get("score_review_json"), dict)
            else {}
        )
        is_retirement_with_played_games = (
            scoring_format == "BEST_2_OF_3"
            and result_type == "RETIREMENT"
            and finalized
            and not parent_result_only
            and parent_review.get("retirement_completed_games_preserved") is True
        )

        if is_finalized_best_of_three and not parent_result_only:
            parent_score_a = _safe_int(parent.get("score_a"))
            parent_score_b = _safe_int(parent.get("score_b"))
            reviewed_game_scores = (
                parent_review.get("game_scores")
                if isinstance(parent_review, dict)
                else None
            )
            looks_like_legacy_aggregate = (
                not series_children
                and not (
                    isinstance(reviewed_game_scores, list)
                    and len(reviewed_game_scores) in {2, 3}
                )
                and parent_score_a is not None
                and parent_score_b is not None
                and max(parent_score_a, parent_score_b) == 2
                and min(parent_score_a, parent_score_b) in {0, 1}
            )
            if looks_like_legacy_aggregate:
                errors.append(
                    _series_validation_error(
                        "BEST_OF_THREE_INDIVIDUAL_GAME_DETAIL_REQUIRED",
                        f"Finalized best-two-of-three matchup {parent_id} stores only the aggregate series result. Its individual game scores cannot be inferred; enter or reconcile the original game-by-game scores before official rating publication.",
                        parent_game_id=parent_id,
                    )
                )
                invalid_parent_ids.add(parent_id)
                continue
            errors.append(
                _series_validation_error(
                    "BEST_OF_THREE_PARENT_NOT_AGGREGATE",
                    f"Finalized best-two-of-three matchup {parent_id} is not marked as an aggregate-only result.",
                    parent_game_id=parent_id,
                )
            )
            invalid_parent_ids.add(parent_id)
            continue
        if parent_result_only and scoring_format != "BEST_2_OF_3":
            errors.append(
                _series_validation_error(
                    "PARENT_RESULT_ONLY_FORMAT_INVALID",
                    f"Aggregate-only matchup {parent_id} is not a best-two-of-three result.",
                    parent_game_id=parent_id,
                )
            )
            invalid_parent_ids.add(parent_id)
            continue
        if parent_result_only and not is_finalized_best_of_three:
            errors.append(
                _series_validation_error(
                    "BEST_OF_THREE_PARENT_NOT_FINALIZED",
                    f"Aggregate-only best-two-of-three matchup {parent_id} is not a finalized played result.",
                    parent_game_id=parent_id,
                )
            )
            invalid_parent_ids.add(parent_id)
            continue

        if not parent_result_only and not is_retirement_with_played_games:
            if series_children:
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_PARENT_INVALID",
                        f"Matchup {parent_id} has series children but is not an aggregate-only best-two-of-three parent.",
                        parent_game_id=parent_id,
                    )
                )
                invalid_parent_ids.add(parent_id)
            elif scoring_format == "BEST_2_OF_3":
                # An unfinished series parent is operational schedule state,
                # never a one-game rating source.  Once finalized, the branch
                # above requires it to become aggregate-only with children.
                continue
            elif _is_rating_publish_eligible(parent):
                rating_entries.append((parent, 0, parent))
            continue

        if not series_children:
            errors.append(
                _series_validation_error(
                    "BEST_OF_THREE_SERIES_GAMES_MISSING",
                    f"Finalized best-two-of-three matchup {parent_id} has no rating-game children.",
                    parent_game_id=parent_id,
                )
            )
            invalid_parent_ids.add(parent_id)
            continue

        numbered_children: list[tuple[int, dict[str, Any]]] = []
        parent_errors_before = len(errors)
        for child in series_children:
            child_id = str(child.get("id") or "")
            game_number = _safe_int(child.get("series_game_number"))
            if (
                str(child.get("stage") or "").strip().upper() != "SERIES_GAME"
                or game_number is None
                or game_number < 1
                or game_number > 3
            ):
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_IDENTITY_INVALID",
                        f"Series child {child_id or '[missing id]'} needs stage SERIES_GAME and a game number from 1 to 3.",
                        parent_game_id=parent_id,
                        child_game_id=child_id,
                    )
                )
                continue
            for field in (
                "tournament_id",
                "draw_id",
                "registration_day_id",
                "event_option_id",
                "team_a_id",
                "team_b_id",
            ):
                if str(child.get(field) or "") != str(parent.get(field) or ""):
                    errors.append(
                        _series_validation_error(
                            "SERIES_GAME_SCOPE_MISMATCH",
                            f"Series child {child_id or '[missing id]'} does not match its parent {field}.",
                            parent_game_id=parent_id,
                            child_game_id=child_id,
                        )
                    )
                    break
            if _tournament_game_scoring_format(child) != "GAME_TO_11":
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_FORMAT_INVALID",
                        f"Series child {child_id or '[missing id]'} must be a game to 11.",
                        parent_game_id=parent_id,
                        child_game_id=child_id,
                    )
                )
            score_a = _safe_int(child.get("score_a"))
            score_b = _safe_int(child.get("score_b"))
            score_review = child.get("score_review_json")
            if (
                not isinstance(score_review, dict)
                or score_review.get("accepted") is not True
                or str(score_review.get("scoring_format") or "").strip().upper()
                != "GAME_TO_11"
                or _safe_int(score_review.get("score_a")) != score_a
                or _safe_int(score_review.get("score_b")) != score_b
            ):
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_REVIEW_INVALID",
                        f"Series child {child_id or '[missing id]'} does not have accepted score-review evidence for its stored score.",
                        parent_game_id=parent_id,
                        child_game_id=child_id,
                    )
                )
            expected_winner = ""
            expected_loser = ""
            if score_a is not None and score_b is not None and score_a != score_b:
                expected_winner = str(
                    parent.get("team_a_id")
                    if score_a > score_b
                    else parent.get("team_b_id")
                    or ""
                )
                expected_loser = str(
                    parent.get("team_b_id")
                    if score_a > score_b
                    else parent.get("team_a_id")
                    or ""
                )
            if (
                not _is_rating_publish_eligible(child)
                or child.get("finalized_at") in (None, "")
                or score_a is None
                or score_b is None
                or score_a < 0
                or score_b < 0
                or score_a == score_b
                or max(score_a, score_b) < 11
                or abs(score_a - score_b) < 2
                or str(child.get("winner_team_id") or "") != expected_winner
                or str(child.get("loser_team_id") or "") != expected_loser
            ):
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_RESULT_INVALID",
                        f"Series child {child_id or '[missing id]'} is not a finalized, non-tied played game with valid winner evidence.",
                        parent_game_id=parent_id,
                        child_game_id=child_id,
                    )
                )
            numbered_children.append((int(game_number), child))

        numbered_children.sort(
            key=lambda item: (item[0], str(item[1].get("id") or ""))
        )
        numbers = [number for number, _child in numbered_children]
        allowed_game_counts = (
            {1, 2} if is_retirement_with_played_games else {2, 3}
        )
        if (
            numbers != list(range(1, len(numbered_children) + 1))
            or len(numbers) not in allowed_game_counts
        ):
            errors.append(
                _series_validation_error(
                    "SERIES_GAME_SEQUENCE_INVALID",
                    (
                        f"Retired best-two-of-three matchup {parent_id} needs one or two uniquely numbered, contiguous completed games."
                        if is_retirement_with_played_games
                        else f"Best-two-of-three matchup {parent_id} needs exactly two or three uniquely numbered, contiguous games."
                    ),
                    parent_game_id=parent_id,
                )
            )
        reviewed_games = (
            parent_review.get("game_scores")
            if isinstance(parent_review, dict)
            else None
        )
        if (
            not isinstance(parent_review, dict)
            or parent_review.get("accepted") is not True
            or str(parent_review.get("scoring_format") or "").strip().upper()
            != "BEST_2_OF_3"
            or _safe_int(parent_review.get("score_a")) != _safe_int(parent.get("score_a"))
            or _safe_int(parent_review.get("score_b")) != _safe_int(parent.get("score_b"))
            or not isinstance(reviewed_games, list)
            or len(reviewed_games) != len(numbered_children)
            or (
                is_retirement_with_played_games
                and (
                    parent_review.get("retirement_completed_games_preserved")
                    is not True
                    or parent_review.get("synthetic_progression_score") is not True
                    or parent_review.get("rating_publish_eligible") is not False
                    or str(parent_review.get("non_playing_team_id") or "")
                    != str(parent.get("loser_team_id") or "")
                )
            )
        ):
            errors.append(
                _series_validation_error(
                    "BEST_OF_THREE_REVIEW_INVALID",
                    f"Best-two-of-three matchup {parent_id} does not have accepted individual-game review evidence for its aggregate result.",
                    parent_game_id=parent_id,
                )
            )
        else:
            for game_number, child in numbered_children:
                reviewed_game = reviewed_games[game_number - 1]
                if (
                    not isinstance(reviewed_game, dict)
                    or _safe_int(reviewed_game.get("game_number")) != game_number
                    or _safe_int(reviewed_game.get("score_a"))
                    != _safe_int(child.get("score_a"))
                    or _safe_int(reviewed_game.get("score_b"))
                    != _safe_int(child.get("score_b"))
                    or reviewed_game.get("score_review")
                    != child.get("score_review_json")
                ):
                    errors.append(
                        _series_validation_error(
                            "SERIES_GAME_REVIEW_MISMATCH",
                            f"Series child {str(child.get('id') or '[missing id]')} does not match its parent review evidence.",
                            parent_game_id=parent_id,
                            child_game_id=str(child.get("id") or ""),
                        )
                    )

        wins_a = 0
        wins_b = 0
        clinched_early = False
        for index, (_number, child) in enumerate(numbered_children):
            if clinched_early:
                errors.append(
                    _series_validation_error(
                        "SERIES_GAME_AFTER_CLINCH",
                        f"Best-two-of-three matchup {parent_id} contains a game after the series was clinched.",
                        parent_game_id=parent_id,
                        child_game_id=str(child.get("id") or ""),
                    )
                )
                break
            score_a = _safe_int(child.get("score_a"))
            score_b = _safe_int(child.get("score_b"))
            if score_a is not None and score_b is not None:
                if score_a > score_b:
                    wins_a += 1
                elif score_b > score_a:
                    wins_b += 1
            clinched_early = (wins_a == 2 or wins_b == 2) and index < len(
                numbered_children
            ) - 1

        parent_score_a = _safe_int(parent.get("score_a"))
        parent_score_b = _safe_int(parent.get("score_b"))
        expected_parent_winner = str(
            parent.get("team_a_id") if wins_a > wins_b else parent.get("team_b_id") or ""
        )
        expected_parent_loser = str(
            parent.get("team_b_id") if wins_a > wins_b else parent.get("team_a_id") or ""
        )
        if is_retirement_with_played_games:
            non_playing_team_id = str(
                parent_review.get("non_playing_team_id") or ""
            )
            parent_winner = str(parent.get("winner_team_id") or "")
            parent_loser = str(parent.get("loser_team_id") or "")
            parent_team_a = str(parent.get("team_a_id") or "")
            parent_team_b = str(parent.get("team_b_id") or "")
            retirement_invalid = (
                max(wins_a, wins_b) >= 2
                or parent_score_a is None
                or parent_score_b is None
                or parent_score_a < 0
                or parent_score_b < 0
                or parent_score_a == parent_score_b
                or parent_loser != non_playing_team_id
                or parent_loser not in {parent_team_a, parent_team_b}
                or parent_winner not in {parent_team_a, parent_team_b}
                or parent_winner == parent_loser
                or (
                    parent_score_a > parent_score_b
                    and parent_winner != parent_team_a
                )
                or (
                    parent_score_b > parent_score_a
                    and parent_winner != parent_team_b
                )
            )
        else:
            retirement_invalid = False
        if retirement_invalid or (
            not is_retirement_with_played_games
            and (
                max(wins_a, wins_b) != 2
                or min(wins_a, wins_b) not in {0, 1}
                or parent_score_a != wins_a
                or parent_score_b != wins_b
                or str(parent.get("winner_team_id") or "")
                != expected_parent_winner
                or str(parent.get("loser_team_id") or "")
                != expected_parent_loser
            )
        ):
            errors.append(
                _series_validation_error(
                    (
                        "RETIREMENT_SERIES_EVIDENCE_INVALID"
                        if is_retirement_with_played_games
                        else "BEST_OF_THREE_AGGREGATE_MISMATCH"
                    ),
                    (
                        f"Retired best-two-of-three matchup {parent_id} has malformed synthetic outcome or completed-game evidence."
                        if is_retirement_with_played_games
                        else f"Best-two-of-three matchup {parent_id} does not match its individual game winners."
                    ),
                    parent_game_id=parent_id,
                )
            )

        if len(errors) != parent_errors_before or parent_id in preinvalid_parent_ids:
            invalid_parent_ids.add(parent_id)
            continue
        for index, (game_number, child) in enumerate(numbered_children):
            rating_child = dict(child)
            rating_child["_series_parent_game"] = dict(parent)
            rating_child["_series_clinching"] = (
                not is_retirement_with_played_games
                and index == len(numbered_children) - 1
            )
            rating_child["_series_game_number"] = game_number
            rating_entries.append((parent, game_number, rating_child))

    for parent_id, series_children in children_by_parent.items():
        if parent_id in competition_game_ids:
            continue
        for child in series_children:
            errors.append(
                _series_validation_error(
                    "SERIES_GAME_PARENT_NOT_FOUND",
                    f"Series child {str(child.get('id') or '[missing id]')} references an unknown or non-parent matchup.",
                    parent_game_id=parent_id,
                    child_game_id=str(child.get("id") or ""),
                )
            )

    rating_entries.sort(
        key=lambda item: (
            _competition_game_sort_key(item[0]),
            item[1],
            str(item[2].get("id") or ""),
        )
    )
    return {
        "competition_games": sorted(
            competition_games, key=_competition_game_sort_key
        ),
        "rating_games": [entry[2] for entry in rating_entries],
        "errors": errors,
        "invalid_parent_game_ids": sorted(invalid_parent_ids),
    }


def _match_exclusion_target_ids(operation: dict[str, Any]) -> set[str]:
    """Return only Match Log ids explicitly bound to an exclusion operation."""

    target_ids = {
        str(value)
        for value in (operation.get("excluded_match_ids") or [])
        if str(value)
    }
    targets = operation.get("targets_json")
    if isinstance(targets, list):
        target_ids.update(
            str(row.get("match_id") or "")
            for row in targets
            if isinstance(row, dict) and str(row.get("match_id") or "")
        )
    return target_ids


def _official_match_mismatch_fields(
    match: dict[str, Any],
    *,
    game: dict[str, Any],
    teams_by_id: dict[str, dict[str, Any]],
    tournament_id: str,
    publication_projection: dict[str, Any] | None,
) -> list[str]:
    """Compare immutable official-link content to its tournament source.

    Source players, scores, and context must remain exact. Match Log
    classification is compared to the completed operation's immutable publish
    projection rather than mutable tournament metadata. Rating replay may
    legitimately update calculated rating columns (and therefore row_version),
    so those derived fields are intentionally outside this immutable link check.
    """

    team_a = teams_by_id.get(str(game.get("team_a_id") or ""))
    team_b = teams_by_id.get(str(game.get("team_b_id") or ""))
    if not team_a or not team_b:
        return ["team_links"]
    singles = team_a.get("player2_id") in (None, "") and team_b.get(
        "player2_id"
    ) in (None, "")
    expected = {
        "tournament_id": str(tournament_id),
        "tournament_game_id": str(game.get("id") or ""),
        "context_type": "tournament_game",
        "context_id": str(game.get("id") or ""),
        "match_format": "singles" if singles else "doubles",
        "t1_p1": _safe_int(team_a.get("player1_id")),
        "t1_p2": _safe_int(team_a.get("player2_id")),
        "t2_p1": _safe_int(team_b.get("player1_id")),
        "t2_p2": _safe_int(team_b.get("player2_id")),
        "score_t1": _safe_int(game.get("score_a")),
        "score_t2": _safe_int(game.get("score_b")),
    }
    actual = {
        "tournament_id": str(match.get("tournament_id") or ""),
        "tournament_game_id": str(match.get("tournament_game_id") or ""),
        "context_type": str(match.get("context_type") or ""),
        "context_id": str(match.get("context_id") or ""),
        "match_format": str(match.get("match_format") or "").strip().lower(),
        "t1_p1": _safe_int(match.get("t1_p1")),
        "t1_p2": _safe_int(match.get("t1_p2")),
        "t2_p1": _safe_int(match.get("t2_p1")),
        "t2_p2": _safe_int(match.get("t2_p2")),
        "score_t1": _safe_int(match.get("score_t1")),
        "score_t2": _safe_int(match.get("score_t2")),
    }
    fields = [field for field, value in expected.items() if actual.get(field) != value]
    if publication_projection is not None:
        expected_classification = {
            "match_type": str(publication_projection.get("match_type") or ""),
            "league": str(publication_projection.get("league") or ""),
            "date": _canonical_publication_date(publication_projection.get("date")),
            "week_tag": str(publication_projection.get("week_tag") or ""),
        }
        actual_classification = {
            "match_type": str(match.get("match_type") or ""),
            "league": str(match.get("league") or ""),
            "date": _canonical_publication_date(match.get("date")),
            "week_tag": str(match.get("week_tag") or ""),
        }
        fields.extend(
            field
            for field in OFFICIAL_MATCH_CLASSIFICATION_FIELDS
            if actual_classification[field] != expected_classification[field]
        )
    if not str(match.get("id") or ""):
        fields.append("id")
    return sorted(set(fields))


def _canonical_publication_date(value: Any) -> str:
    parsed = coerce_utc_datetime(value)
    return parsed.isoformat() if parsed is not None else str(value or "")


def _official_publication_plan_evidence(
    operations: list[dict[str, Any]],
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, dict[str, Any]]:
    """Read exact match projections from completed guarded publish intents.

    The operation request is persisted before mutation and the atomic publish
    core compares its exact publish plan before inserting Match Log rows. A
    completed row is therefore the existing publication-time authority; the
    mutable tournament/draw labels are not safe substitutes during closeout.
    """

    evidence: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if (
            str(operation.get("action") or "") not in OFFICIAL_PUBLISH_ACTIONS
            or str(operation.get("status") or "").strip().lower() != "completed"
        ):
            continue
        draw_id = str(operation.get("entity_id") or "")
        state = evidence.setdefault(
            draw_id,
            {
                "operation_keys": [],
                "errors": [],
                "valid_plans": [],
            },
        )
        operation_key = str(operation.get("operation_key") or "")
        state["operation_keys"].append(operation_key)
        request = operation.get("request_json")
        if not isinstance(request, dict):
            state["errors"].append("PUBLISH_OPERATION_REQUEST_MISSING")
            continue
        identity_fields = (
            "operation_key",
            "request_fingerprint",
            "club_id",
            "surface",
            "action",
            "entity_type",
            "entity_id",
            "lock_scope",
            "expected_state",
        )
        if any(
            str(request.get(field) or "") != str(operation.get(field) or "")
            for field in identity_fields
        ):
            state["errors"].append("PUBLISH_OPERATION_IDENTITY_MISMATCH")
            continue
        payload = request.get("payload")
        if not isinstance(payload, dict):
            state["errors"].append("PUBLISH_OPERATION_PAYLOAD_INVALID")
            continue
        request_body = {
            "club_id": str(request.get("club_id") or "").strip(),
            "surface": str(request.get("surface") or "").strip(),
            "action": str(request.get("action") or "").strip(),
            "entity_type": str(request.get("entity_type") or "").strip(),
            "entity_id": str(request.get("entity_id") or "").strip(),
            "lock_scope": str(request.get("lock_scope") or "").strip(),
            "expected_state": str(request.get("expected_state") or "").strip(),
            "payload": dict(payload),
        }
        if str(request.get("idempotency_key") or "").strip():
            request_body["idempotency_key"] = str(
                request.get("idempotency_key") or ""
            ).strip()
        expected_request_fingerprint = stable_tournament_admin_fingerprint(request_body)
        expected_operation_key = stable_tournament_admin_fingerprint(
            {
                "contract": "jupr:tournament-admin:v1",
                "request_fingerprint": expected_request_fingerprint,
            }
        )
        if (
            expected_request_fingerprint != str(operation.get("request_fingerprint") or "")
            or expected_operation_key != operation_key
            or str(operation.get("club_id") or "") != str(club_id)
            or str(operation.get("entity_type") or "") != "tournament_event_draw"
        ):
            state["errors"].append("PUBLISH_OPERATION_IDENTITY_INVALID")
            continue
        plan = payload.get("publish_plan")
        if not isinstance(plan, dict):
            state["errors"].append("PUBLISH_PLAN_MISSING")
            continue
        tournament_metadata = plan.get("tournament_metadata")
        projections = plan.get("match_payload_projections")
        declared_ids = [
            str(value)
            for value in (plan.get("tournament_game_ids") or [])
            if str(value)
        ]
        if (
            str(plan.get("draw_id") or "") != draw_id
            or not isinstance(tournament_metadata, dict)
            or str(tournament_metadata.get("id") or "") != str(tournament_id)
            or not isinstance(projections, list)
            or not projections
            or not declared_ids
            or len(declared_ids) != len(set(declared_ids))
            or int(_safe_int(plan.get("match_count")) or -1) != len(declared_ids)
            or any(not isinstance(row, dict) for row in projections)
        ):
            state["errors"].append("PUBLISH_PLAN_SCOPE_INVALID")
            continue
        projection_ids = [
            str(row.get("tournament_game_id") or "")
            for row in projections
            if isinstance(row, dict)
        ]
        if (
            len(projection_ids) != len(declared_ids)
            or len(projection_ids) != len(set(projection_ids))
            or set(projection_ids) != set(declared_ids)
            or any(
                str(row.get("tournament_id") or "") != str(tournament_id)
                for row in projections
                if isinstance(row, dict)
            )
        ):
            state["errors"].append("PUBLISH_PLAN_GAME_SET_INVALID")
            continue
        expected_fingerprints = sorted(
            [
                {
                    "tournament_game_id": str(row.get("tournament_game_id") or ""),
                    "payload_fingerprint": stable_tournament_admin_fingerprint(row),
                }
                for row in projections
                if isinstance(row, dict)
            ],
            key=lambda row: row["tournament_game_id"],
        )
        if plan.get("match_payload_fingerprints") != expected_fingerprints:
            state["errors"].append("PUBLISH_PLAN_FINGERPRINT_INVALID")
            continue
        state["valid_plans"].append(
            {
                "operation_key": operation_key,
                "game_ids": sorted(declared_ids),
                "projections": {
                    str(row.get("tournament_game_id") or ""): dict(row)
                    for row in projections
                    if isinstance(row, dict)
                },
            }
        )

    result: dict[str, dict[str, Any]] = {}
    for draw_id, state in evidence.items():
        valid_plans = list(state.get("valid_plans") or [])
        errors = list(state.get("errors") or [])
        if len(valid_plans) > 1:
            errors.append("MULTIPLE_COMPLETED_PUBLISH_PLANS")
        plan = valid_plans[0] if len(valid_plans) == 1 and not errors else {}
        result[draw_id] = {
            "available": bool(plan),
            "operation_keys": sorted(
                str(value) for value in state.get("operation_keys") or [] if str(value)
            ),
            "errors": sorted(set(str(value) for value in errors if str(value))),
            "game_ids": list(plan.get("game_ids") or []),
            "projections": dict(plan.get("projections") or {}),
        }
    return result


def _blocker(
    code: str,
    message: str,
    *,
    scope: str = "tournament",
    draw_id: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    blocker: dict[str, Any] = {
        "code": str(code),
        "scope": str(scope),
        "message": str(message),
    }
    if draw_id:
        blocker["draw_id"] = str(draw_id)
    if entity_type:
        blocker["entity_type"] = str(entity_type)
    if entity_id:
        blocker["entity_id"] = str(entity_id)
    if count is not None:
        blocker["count"] = int(count)
    return blocker


def _dedupe_blockers(blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for blocker in blockers:
        key = (
            str(blocker.get("code") or ""),
            str(blocker.get("draw_id") or ""),
            str(blocker.get("entity_id") or ""),
            str(blocker.get("message") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(blocker)
    return result


def _dedupe_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one authoritative row per durable operation identity."""

    by_key: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in operations:
        operation_key = str(row.get("operation_key") or "").strip()
        if not operation_key:
            anonymous.append(row)
            continue
        current = by_key.get(operation_key)
        if current is None or str(row.get("updated_at") or row.get("created_at") or "") >= str(
            current.get("updated_at") or current.get("created_at") or ""
        ):
            by_key[operation_key] = row
    return [*by_key.values(), *anonymous]


def _readiness(blockers: list[dict[str, Any]], *, complete: bool = False) -> dict[str, Any]:
    blockers = _dedupe_blockers(blockers)
    return {
        "ready": not blockers and not complete,
        "complete": bool(complete),
        "state": "complete" if complete else "ready" if not blockers else "blocked",
        "blockers": blockers,
    }


def _team_label(team: dict[str, Any], player_names: dict[int, str]) -> tuple[str, list[str]]:
    names: list[str] = []
    for raw_player_id in (team.get("player1_id"), team.get("player2_id")):
        player_id = _safe_int(raw_player_id)
        if player_id is not None:
            names.append(player_names.get(player_id, f"Player {player_id}"))
    fallback = f"Team {int(_safe_int(team.get('team_number')) or 0)}"
    return " / ".join(names) or fallback, names


def _expected_awards(
    *,
    tournament_id: str,
    draw_id: str,
    teams: list[dict[str, Any]],
    podium: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    teams_by_id = {str(row.get("id") or ""): row for row in teams}
    expected: list[dict[str, Any]] = []
    for podium_row in podium:
        placement = _safe_int(podium_row.get("placement"))
        badge_id = PODIUM_BADGE_MAP.get(int(placement or 0))
        team = teams_by_id.get(str(podium_row.get("team_id") or ""))
        if not badge_id or not team:
            continue
        context_id = f"{tournament_id}:draw:{draw_id}:podium:{placement}"
        for raw_player_id in (team.get("player1_id"), team.get("player2_id")):
            player_id = _safe_int(raw_player_id)
            if player_id is not None:
                expected.append(
                    {
                        "player_id": player_id,
                        "badge_id": str(badge_id),
                        "context_id": context_id,
                    }
                )
    return sorted(
        expected,
        key=lambda row: (str(row["context_id"]), str(row["badge_id"]), int(row["player_id"])),
    )


def _award_key_sets(
    expected_awards: list[dict[str, Any]],
    award_rows: list[dict[str, Any]],
) -> tuple[set[tuple[int | None, str, str]], set[tuple[int | None, str, str]]]:
    expected = {
        (_safe_int(row.get("player_id")), str(row.get("badge_id") or ""), str(row.get("context_id") or ""))
        for row in expected_awards
    }
    actual = {
        (_safe_int(row.get("player_id")), str(row.get("badge_id") or ""), str(row.get("context_id") or ""))
        for row in award_rows
        if not row.get("revoked_at")
    }
    return expected, actual


def _draw_core_blockers(
    *,
    draw_id: str,
    draw_name: str,
    teams: list[dict[str, Any]],
    games: list[dict[str, Any]],
    podium: list[dict[str, Any]],
    review: dict[str, Any],
    expected_awards: list[dict[str, Any]],
    award_rows: list[dict[str, Any]],
    awards_available: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    team_ids = {str(row.get("id") or "") for row in teams if row.get("id")}
    invalid_games = [game for game in games if not _is_finalized_game(game, team_ids=team_ids)]
    finalized_games = len(games) - len(invalid_games)
    tied_games = [
        game
        for game in games
        if _safe_int(game.get("score_a")) is not None
        and _safe_int(game.get("score_a")) == _safe_int(game.get("score_b"))
    ]
    if teams and not games:
        blockers.append(
            _blocker(
                "DRAW_GAMES_MISSING",
                f"{draw_name} has {len(teams)} teams but no tournament games.",
                scope="draw",
                draw_id=draw_id,
                entity_type="tournament_event_draw",
                entity_id=draw_id,
                count=len(teams),
            )
        )
    elif not games:
        blockers.append(
            _blocker(
                "DRAW_GAMES_MISSING",
                f"{draw_name} has no tournament games.",
                scope="draw",
                draw_id=draw_id,
                entity_type="tournament_event_draw",
                entity_id=draw_id,
                count=0,
            )
        )
    if invalid_games:
        blockers.append(
            _blocker(
                "GAMES_NOT_FINALIZED",
                f"{draw_name} has {len(invalid_games)} game(s) without a finalized, non-tied score and valid team/winner evidence.",
                scope="draw",
                draw_id=draw_id,
                entity_type="tournament_game",
                count=len(invalid_games),
            )
        )

    podium_rows = sorted(
        [
            {
                "placement": _safe_int(row.get("placement")),
                "team_id": str(row.get("team_id") or ""),
                "source": str(row.get("source") or "").upper(),
            }
            for row in podium
        ],
        key=lambda row: int(row.get("placement") or 0),
    )
    placements = [row["placement"] for row in podium_rows]
    podium_team_ids = [str(row["team_id"]) for row in podium_rows]
    podium_complete = (
        placements == [1, 2, 3]
        and len(set(podium_team_ids)) == 3
        and all(team_id in team_ids for team_id in podium_team_ids)
    )
    if not podium_complete:
        blockers.append(
            _blocker(
                "PODIUM_INCOMPLETE",
                f"{draw_name} needs exactly one valid first-, second-, and third-place team.",
                scope="draw",
                draw_id=draw_id,
                entity_type="tournament_podium",
                count=len(podium_rows),
            )
        )
    if not bool(review.get("current")):
        blockers.append(
            _blocker(
                "PODIUM_REVIEW_REQUIRED",
                str((review.get("blockers") or ["The current podium has not been explicitly reviewed."])[0]),
                scope="draw",
                draw_id=draw_id,
                entity_type="tournament_event_draw",
                entity_id=draw_id,
            )
        )

    expected_keys, awarded_keys = _award_key_sets(expected_awards, award_rows)
    active_award_rows = [row for row in award_rows if not row.get("revoked_at")]
    duplicate_award_count = max(0, len(active_award_rows) - len(awarded_keys))
    duplicate_awards = duplicate_award_count > 0
    unexpected_awards = len(awarded_keys - expected_keys)
    awards_complete = bool(expected_keys) and awarded_keys == expected_keys and not duplicate_awards
    if not awards_available:
        blockers.append(
            _blocker(
                "AWARD_EVIDENCE_UNAVAILABLE",
                f"{draw_name} podium award evidence is unavailable.",
                scope="draw",
                draw_id=draw_id,
            )
        )
    elif not expected_keys:
        blockers.append(
            _blocker(
                "AWARDS_NOT_DERIVABLE",
                f"{draw_name} has no exact linked-player award set for its podium.",
                scope="draw",
                draw_id=draw_id,
            )
        )
    elif awarded_keys != expected_keys or duplicate_awards:
        missing = len(expected_keys - awarded_keys)
        unexpected = len(awarded_keys - expected_keys)
        blockers.append(
            _blocker(
                "AWARDS_INCOMPLETE",
                f"{draw_name} podium awards are not exact ({missing} missing, {unexpected} unexpected).",
                scope="draw",
                draw_id=draw_id,
                entity_type="player_badge",
                count=missing + unexpected + (1 if duplicate_awards else 0),
            )
        )

    return blockers, {
        "games": len(games),
        "finalized_games": finalized_games,
        "open_games": len(invalid_games),
        "invalid_games": len(invalid_games),
        "tied_games": len(tied_games),
        "podium_entries": len(podium_rows),
        "podium_complete": podium_complete,
        "podium_reviewed": bool(review.get("current")),
        "expected_awards": len(expected_keys),
        "verified_awards": len(expected_keys.intersection(awarded_keys)),
        "active_awards": len(awarded_keys),
        "unexpected_awards": unexpected_awards,
        "duplicate_awards": duplicate_award_count,
        "awards_complete": awards_complete,
    }


def build_admin_tournament_lifecycle(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    selected_draw_id: str | None = None,
    runtime_capability: dict[str, Any] | None = None,
    ignore_operation_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_selected_draw_id = _clean_text(selected_draw_id, limit=120) or None
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    warnings: list[str] = []
    draws_all, draws_available, warning = _read_rows(
        supabase,
        "tournament_event_draws",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)
    teams_all, teams_available, warning = _read_rows(
        supabase,
        "tournament_teams",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)
    games_all, games_available, warning = _read_rows(
        supabase,
        "tournament_games",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)
    podium_all, podium_available, warning = _read_rows(
        supabase,
        "tournament_podium",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)
    players, players_available, warning = _read_rows(
        supabase,
        "players",
        filters=(("club_id", str(club_id)),),
    )
    if warning:
        warnings.append(warning)
    event_options, event_options_available, warning = _read_rows(
        supabase,
        "tournament_event_options",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)
    match_rows_all, matches_available, warning = _read_rows(
        supabase,
        "matches",
        filters=(("club_id", str(club_id)), ("tournament_id", clean_tournament_id)),
    )
    if warning:
        warnings.append(warning)
    badge_rows_all, awards_available, warning = _read_rows(
        supabase,
        "player_badges",
        filters=(("club_id", str(club_id)), ("context_type", "tournament")),
    )
    if warning:
        warnings.append(warning)
    operations, operations_available, warning = _read_rows(
        supabase,
        "tournament_admin_operations",
        filters=(("club_id", str(club_id)),),
    )
    if warning:
        warnings.append(warning)
    match_exclusion_operations, match_exclusions_available, warning = _read_rows(
        supabase,
        "match_exclusion_operations",
        filters=(("club_id", str(club_id)),),
        in_filters=(("status", tuple(sorted(UNSETTLED_MATCH_EXCLUSION_STATUSES))),),
    )
    if warning:
        warnings.append(warning)
    # Replay jobs are read only after the tournament's linked exclusions are
    # known. Starting with an empty, available evidence set avoids a club-wide
    # history scan for tournaments that have no linked exclusion recovery.
    replay_jobs: list[dict[str, Any]] = []
    replay_jobs_available = True
    day_live_runs, day_live_runs_available, warning = _read_rows(
        supabase,
        "tournament_day_live_runs",
        filters=(("tournament_id", clean_tournament_id),),
    )
    if warning:
        warnings.append(warning)

    global_evidence_blockers: list[dict[str, Any]] = []
    for available, code, message in (
        (draws_available, "DRAW_EVIDENCE_UNAVAILABLE", "Tournament draw evidence is unavailable."),
        (teams_available, "TEAM_EVIDENCE_UNAVAILABLE", "Tournament team evidence is unavailable."),
        (games_available, "GAME_EVIDENCE_UNAVAILABLE", "Tournament game evidence is unavailable."),
        (podium_available, "PODIUM_EVIDENCE_UNAVAILABLE", "Tournament podium evidence is unavailable."),
        (
            event_options_available,
            "EVENT_OPTION_EVIDENCE_UNAVAILABLE",
            "Tournament event-option evidence is unavailable.",
        ),
        (matches_available, "OFFICIAL_LINK_EVIDENCE_UNAVAILABLE", "Official Match Log links are unavailable."),
        (awards_available, "AWARD_EVIDENCE_UNAVAILABLE", "Podium award evidence is unavailable."),
        (operations_available, "OPERATION_EVIDENCE_UNAVAILABLE", "Tournament operation evidence is unavailable."),
        (
            day_live_runs_available,
            "DAY_LIVE_EVIDENCE_UNAVAILABLE",
            "Tournament day-live closeout evidence is unavailable.",
        ),
    ):
        if not available:
            global_evidence_blockers.append(_blocker(code, message))

    all_draw_ids = {
        str(row.get("id") or "") for row in draws_all if row.get("id")
    }
    primary_draws = [row for row in draws_all if _is_active_primary_draw(row)]
    primary_draw_ids = {str(row.get("id") or "") for row in primary_draws if row.get("id")}
    active_team_parent_draws = [
        row
        for row in draws_all
        if str(row.get("draw_kind") or "").strip().upper() == "TEAM_PARENT"
        and str(row.get("status") or "draft").strip().lower()
        not in INACTIVE_DRAW_STATUSES
    ]
    active_primary_event_option_ids = {
        str(row.get("event_option_id") or "")
        for row in primary_draws
        if str(row.get("event_option_id") or "")
    }
    enabled_event_options_without_draw = [
        row
        for row in event_options
        if row.get("id")
        and _enabled(row.get("enabled"), default=True)
        and str(row.get("status") or "active").strip().lower()
        not in INACTIVE_DRAW_STATUSES
        and str(row.get("id") or "") not in active_primary_event_option_ids
    ]
    for event_option in enabled_event_options_without_draw:
        event_id = str(event_option.get("id") or "")
        event_name = str(
            event_option.get("division_name")
            or event_option.get("label")
            or event_option.get("event_family_label")
            or "Enabled event"
        )
        global_evidence_blockers.append(
            _blocker(
                "EVENT_DRAW_MISSING",
                f"{event_name} is enabled but has no active tournament draw. Create its draw or explicitly cancel the empty event.",
                scope="event",
                entity_type="tournament_event_option",
                entity_id=event_id,
            )
        )
    inactive_primary_draw_ids = {
        str(row.get("id") or "")
        for row in draws_all
        if row.get("id") and _is_primary_draw(row) and not _is_active_primary_draw(row)
    }
    if clean_selected_draw_id and clean_selected_draw_id not in primary_draw_ids:
        non_active = next(
            (row for row in draws_all if str(row.get("id") or "") == clean_selected_draw_id),
            None,
        )
        if non_active is None:
            raise ValueError("draw not found for this tournament")
        if _is_primary_draw(non_active):
            raise ValueError("This tournament draw is inactive and cannot enter closeout.")

    for team_parent in active_team_parent_draws:
        team_parent_id = str(team_parent.get("id") or "")
        global_evidence_blockers.append(
            _blocker(
                "TEAM_COMPETITION_CLOSEOUT_UNSUPPORTED",
                "Four-player team competition closeout is blocked until its podium has canonical explicit review and exact award evidence.",
                scope="draw",
                draw_id=team_parent_id,
                entity_type="tournament_event_draw",
                entity_id=team_parent_id,
            )
        )

    orphan_teams = [
        row
        for row in teams_all
        if not str(row.get("draw_id") or "")
        or str(row.get("draw_id") or "") not in all_draw_ids
    ]
    inactive_draw_teams = [
        row
        for row in teams_all
        if str(row.get("draw_id") or "") in inactive_primary_draw_ids
    ]
    orphan_games = [
        row
        for row in games_all
        if not str(row.get("draw_id") or "")
        or str(row.get("draw_id") or "") not in all_draw_ids
    ]
    inactive_draw_games = [
        row
        for row in games_all
        if str(row.get("draw_id") or "") in inactive_primary_draw_ids
    ]
    orphan_podium = [
        row
        for row in podium_all
        if not str(row.get("draw_id") or "")
        or str(row.get("draw_id") or "") not in all_draw_ids
    ]
    inactive_draw_podium = [
        row
        for row in podium_all
        if str(row.get("draw_id") or "") in inactive_primary_draw_ids
    ]
    if orphan_teams:
        global_evidence_blockers.append(
            _blocker(
                "ORPHAN_TOURNAMENT_TEAMS",
                f"{len(orphan_teams)} tournament team row(s) are not linked to a known draw and require reconciliation.",
                entity_type="tournament_team",
                count=len(orphan_teams),
            )
        )
    if inactive_draw_teams:
        global_evidence_blockers.append(
            _blocker(
                "OUT_OF_SCOPE_TOURNAMENT_TEAMS",
                f"{len(inactive_draw_teams)} tournament team row(s) belong to an inactive draw and require reconciliation.",
                entity_type="tournament_team",
                count=len(inactive_draw_teams),
            )
        )
    if orphan_games:
        global_evidence_blockers.append(
            _blocker(
                "ORPHAN_TOURNAMENT_GAMES",
                f"{len(orphan_games)} tournament game row(s) are not linked to a known draw and require reconciliation.",
                entity_type="tournament_game",
                count=len(orphan_games),
            )
        )
    if inactive_draw_games:
        global_evidence_blockers.append(
            _blocker(
                "OUT_OF_SCOPE_TOURNAMENT_GAMES",
                f"{len(inactive_draw_games)} tournament game row(s) belong to an inactive draw and require reconciliation.",
                entity_type="tournament_game",
                count=len(inactive_draw_games),
            )
        )
    if orphan_podium:
        global_evidence_blockers.append(
            _blocker(
                "ORPHAN_TOURNAMENT_PODIUM",
                f"{len(orphan_podium)} tournament podium row(s) are not linked to a known draw and require reconciliation.",
                entity_type="tournament_podium",
                count=len(orphan_podium),
            )
        )
    if inactive_draw_podium:
        global_evidence_blockers.append(
            _blocker(
                "OUT_OF_SCOPE_TOURNAMENT_PODIUM",
                f"{len(inactive_draw_podium)} tournament podium row(s) belong to an inactive draw and require reconciliation.",
                entity_type="tournament_podium",
                count=len(inactive_draw_podium),
            )
        )

    teams_by_draw: dict[str, list[dict[str, Any]]] = {draw_id: [] for draw_id in primary_draw_ids}
    games_by_draw: dict[str, list[dict[str, Any]]] = {draw_id: [] for draw_id in primary_draw_ids}
    podium_by_draw: dict[str, list[dict[str, Any]]] = {draw_id: [] for draw_id in primary_draw_ids}
    for row in teams_all:
        teams_by_draw.setdefault(str(row.get("draw_id") or ""), []).append(row)
    for row in games_all:
        games_by_draw.setdefault(str(row.get("draw_id") or ""), []).append(row)
    for row in podium_all:
        podium_by_draw.setdefault(str(row.get("draw_id") or ""), []).append(row)

    # Every non-protected active draw is lifecycle truth, including an empty
    # draw that has not yet been initialized. Omitting those rows would make a
    # partially configured division disappear from tournament-wide closeout.
    participating_draws = list(primary_draws)
    participating_ids = {str(row.get("id") or "") for row in participating_draws}
    relevant_game_ids = {
        str(row.get("id") or "")
        for row in games_all
        if str(row.get("draw_id") or "") in participating_ids and row.get("id")
    }
    relevant_match_rows = [
        row
        for row in match_rows_all
        if str(row.get("tournament_game_id") or "") in relevant_game_ids
    ]
    tournament_match_ids = {
        str(row.get("id") or "") for row in relevant_match_rows if row.get("id")
    }
    relevant_match_exclusion_operations = [
        row
        for row in match_exclusion_operations
        if _match_exclusion_target_ids(row).intersection(tournament_match_ids)
    ]
    relevant_replay_job_ids = {
        str(row.get("replay_job_id") or "")
        for row in relevant_match_exclusion_operations
        if str(row.get("replay_job_id") or "")
    }
    if relevant_replay_job_ids:
        replay_jobs, replay_jobs_available, warning = _read_rows(
            supabase,
            "replay_jobs",
            filters=(("club_id", str(club_id)),),
            in_filters=(
                ("id", tuple(sorted(relevant_replay_job_ids))),
                ("status", tuple(sorted(UNSETTLED_REPLAY_JOB_STATUSES))),
            ),
        )
        if warning:
            warnings.append(warning)
    if tournament_match_ids and not match_exclusions_available:
        global_evidence_blockers.append(
            _blocker(
                "MATCH_EXCLUSION_EVIDENCE_UNAVAILABLE",
                "Tournament-linked Match Log exclusion recovery evidence is unavailable.",
            )
        )
    if relevant_replay_job_ids and not replay_jobs_available:
        global_evidence_blockers.append(
            _blocker(
                "MATCH_REPLAY_EVIDENCE_UNAVAILABLE",
                "Tournament-linked Match Log replay evidence is unavailable.",
            )
        )
    soft_deleted_matches = [
        row for row in relevant_match_rows if row.get("deleted_at") not in (None, "")
    ]
    matches = [
        row for row in relevant_match_rows if row.get("deleted_at") in (None, "")
    ]
    if soft_deleted_matches:
        global_evidence_blockers.append(
            _blocker(
                "OFFICIAL_MATCH_HISTORY_EXCLUDED",
                f"{len(soft_deleted_matches)} tournament-linked Match Log row(s) are soft-deleted and require explicit reconciliation; they may not be republished.",
                entity_type="match",
                count=len(soft_deleted_matches),
            )
        )
    matches_by_game: dict[str, list[dict[str, Any]]] = {}
    for row in matches:
        matches_by_game.setdefault(str(row.get("tournament_game_id") or ""), []).append(row)

    operations = _dedupe_operations(
        [
            row
            for row in operations
            if (
                str(row.get("lock_scope") or "") == clean_tournament_id
                or str(row.get("lock_scope") or "").startswith(
                    f"tournament:{clean_tournament_id}:"
                )
            )
        ]
    )
    publication_plan_evidence_by_draw = _official_publication_plan_evidence(
        operations,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
    )
    ignored_operation_keys = {
        str(value) for value in (ignore_operation_keys or []) if str(value)
    }
    active_operations = [
        row
        for row in operations
        if str(row.get("status") or "").lower() in ACTIVE_OPERATION_STATUSES
        and str(row.get("operation_key") or "") not in ignored_operation_keys
    ]
    recovery_operations = [
        row
        for row in operations
        if str(row.get("status") or "").lower() == "recovery_required"
        and str(row.get("operation_key") or "") not in ignored_operation_keys
    ]
    uncertain_operations = [
        row
        for row in operations
        if str(row.get("status") or "").lower() in UNCERTAIN_OPERATION_STATUSES
        and str(row.get("operation_key") or "") not in ignored_operation_keys
    ]
    operation_blockers: list[dict[str, Any]] = []
    if active_operations:
        operation_blockers.append(
            _blocker(
                "ACTIVE_OR_UNCERTAIN_OPERATIONS",
                f"{len(active_operations)} tournament operation(s) are active or require reconciliation.",
                entity_type="tournament_admin_operation",
                count=len(active_operations),
            )
        )
    unsettled_match_exclusions = [
        row
        for row in relevant_match_exclusion_operations
        if str(row.get("status") or "").strip().lower()
        in UNSETTLED_MATCH_EXCLUSION_STATUSES
    ]
    unsettled_replay_jobs = [
        row
        for row in replay_jobs
        if str(row.get("id") or "") in relevant_replay_job_ids
        and str(row.get("status") or "").strip().lower()
        in UNSETTLED_REPLAY_JOB_STATUSES
    ]
    if unsettled_match_exclusions:
        operation_blockers.append(
            _blocker(
                "MATCH_EXCLUSION_RECOVERY_UNSETTLED",
                f"{len(unsettled_match_exclusions)} tournament-linked Match Log exclusion operation(s) are not fully reconciled.",
                entity_type="match_exclusion_operation",
                count=len(unsettled_match_exclusions),
            )
        )
    if unsettled_replay_jobs:
        operation_blockers.append(
            _blocker(
                "MATCH_REPLAY_UNSETTLED",
                f"{len(unsettled_replay_jobs)} tournament-linked rating replay job(s) are pending or running.",
                entity_type="replay_job",
                count=len(unsettled_replay_jobs),
            )
        )
    active_day_live_runs = [
        row
        for row in day_live_runs
        if str(row.get("state") or "").strip().upper() in {"ACTIVE", "PAUSED"}
    ]
    if active_day_live_runs:
        operation_blockers.append(
            _blocker(
                "DAY_LIVE_RUN_OPEN",
                f"{len(active_day_live_runs)} tournament day-live run(s) are still active or paused. Close them before tournament completion.",
                entity_type="tournament_day_live_run",
                count=len(active_day_live_runs),
            )
        )

    player_names = {
        int(player_id): str(row.get("name") or f"Player {player_id}")
        for row in players
        if (player_id := _safe_int(row.get("id"))) is not None
    }
    event_options_by_id = {
        str(row.get("id") or ""): row for row in event_options if row.get("id")
    }

    draw_models: list[dict[str, Any]] = []
    core_blockers_by_draw: dict[str, list[dict[str, Any]]] = {}
    publication_state_by_draw: dict[str, str] = {}
    all_core_blockers: list[dict[str, Any]] = list(global_evidence_blockers)
    total_expected_awards = 0
    total_verified_awards = 0
    total_finalized_games = 0
    total_open_games = 0
    total_tied_games = 0
    total_podium_entries = 0
    total_podium_reviewed = 0
    total_unexpected_awards = 0
    verified_published_game_ids: set[str] = set()
    official_match_payload_mismatches: list[dict[str, Any]] = []
    official_matches_without_publication_evidence: list[dict[str, Any]] = []

    for draw in sorted(
        participating_draws,
        key=lambda row: (str(row.get("name") or ""), str(row.get("id") or "")),
    ):
        draw_id = str(draw.get("id") or "")
        option = event_options_by_id.get(str(draw.get("event_option_id") or ""), {})
        draw_name = str(
            draw.get("name")
            or option.get("division_name")
            or option.get("event_family_label")
            or "Tournament draw"
        )
        draw_teams = sorted(
            teams_by_draw.get(draw_id, []),
            key=lambda row: (int(_safe_int(row.get("team_number")) or 0), str(row.get("id") or "")),
        )
        draw_games = sorted(
            games_by_draw.get(draw_id, []),
            key=lambda row: (
                str(row.get("stage") or ""),
                int(_safe_int(row.get("rr_round_number")) or 0),
                int(_safe_int(row.get("rr_slot_number")) or 0),
                str(row.get("playoff_game_code") or ""),
                str(row.get("id") or ""),
            ),
        )
        draw_podium = sorted(
            podium_by_draw.get(draw_id, []),
            key=lambda row: (int(_safe_int(row.get("placement")) or 0), str(row.get("id") or "")),
        )
        review_fingerprint = build_admin_tournament_podium_review_fingerprint(
            draw=draw,
            teams=draw_teams,
            games=draw_games,
            podium=draw_podium,
        )
        review = find_current_admin_tournament_podium_review(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=draw_id,
            review_fingerprint=review_fingerprint,
        )
        if not bool(review.get("available")):
            all_core_blockers.append(
                _blocker(
                    "PODIUM_REVIEW_EVIDENCE_UNAVAILABLE",
                    f"{draw_name} podium review evidence is unavailable.",
                    scope="draw",
                    draw_id=draw_id,
                )
            )

        expected_awards = _expected_awards(
            tournament_id=clean_tournament_id,
            draw_id=draw_id,
            teams=draw_teams,
            podium=draw_podium,
        )
        context_prefix = f"{clean_tournament_id}:draw:{draw_id}:podium:"
        award_rows = [
            row
            for row in badge_rows_all
            if str(row.get("context_id") or "").startswith(context_prefix)
        ]
        rating_game_plan = build_tournament_rating_game_plan(draw_games)
        competition_games = list(rating_game_plan["competition_games"])
        series_errors = list(rating_game_plan["errors"])
        draw_core_blockers, draw_counts = _draw_core_blockers(
            draw_id=draw_id,
            draw_name=draw_name,
            teams=draw_teams,
            games=competition_games,
            podium=draw_podium,
            review=review,
            expected_awards=expected_awards,
            award_rows=award_rows,
            awards_available=awards_available,
        )
        if series_errors:
            missing_legacy_details = [
                row
                for row in series_errors
                if str(row.get("code") or "")
                == "BEST_OF_THREE_INDIVIDUAL_GAME_DETAIL_REQUIRED"
            ]
            if missing_legacy_details:
                draw_core_blockers.append(
                    _blocker(
                        "BEST_OF_THREE_INDIVIDUAL_GAME_DETAIL_REQUIRED",
                        f"{draw_name} has {len(missing_legacy_details)} finalized best-two-of-three matchup(s) that store only aggregate series results. Individual game scores cannot be reconstructed; reconcile the original game-by-game scores before official rating publication.",
                        scope="draw",
                        draw_id=draw_id,
                        entity_type="tournament_game",
                        count=len(missing_legacy_details),
                    )
                )
            other_series_errors = [
                row for row in series_errors if row not in missing_legacy_details
            ]
            if other_series_errors:
                draw_core_blockers.append(
                    _blocker(
                        "BEST_OF_THREE_RATING_SOURCE_INVALID",
                        f"{draw_name} has {len(other_series_errors)} invalid best-two-of-three rating-source condition(s). Correct or reconcile its individual game rows before official publication.",
                        scope="draw",
                        draw_id=draw_id,
                        entity_type="tournament_game",
                        count=len(other_series_errors),
                    )
                )
        core_blockers_by_draw[draw_id] = draw_core_blockers
        all_core_blockers.extend(draw_core_blockers)

        draw_game_ids = [str(row.get("id") or "") for row in draw_games if row.get("id")]
        draw_game_id_set = set(draw_game_ids)
        rating_publish_game_ids = [
            str(row.get("id") or "")
            for row in rating_game_plan["rating_games"]
            if row.get("id")
        ]
        rating_publish_game_id_set = set(rating_publish_game_ids)
        draw_matches = [
            row
            for row in matches
            if str(row.get("tournament_game_id") or "") in draw_game_id_set
        ]
        draw_games_by_id = {
            str(row.get("id") or ""): row for row in draw_games if row.get("id")
        }
        draw_teams_by_id = {
            str(row.get("id") or ""): row for row in draw_teams if row.get("id")
        }
        immutable_plan = dict(publication_plan_evidence_by_draw.get(draw_id) or {})
        immutable_plan_errors = list(immutable_plan.get("errors") or [])
        if bool(immutable_plan.get("available")) and set(
            str(value) for value in immutable_plan.get("game_ids") or []
        ) != rating_publish_game_id_set:
            immutable_plan_errors.append("PUBLISH_PLAN_CURRENT_GAME_SET_MISMATCH")
        immutable_plan_available = bool(immutable_plan.get("available")) and not immutable_plan_errors
        immutable_projections = (
            dict(immutable_plan.get("projections") or {})
            if immutable_plan_available
            else {}
        )
        mismatched_matches: list[dict[str, Any]] = []
        mismatched_game_ids: set[str] = set()
        missing_publication_evidence: list[dict[str, str]] = []
        for match in draw_matches:
            game_id = str(match.get("tournament_game_id") or "")
            publication_projection = immutable_projections.get(game_id)
            if publication_projection is None:
                missing = {
                    "match_id": str(match.get("id") or ""),
                    "tournament_game_id": game_id,
                }
                missing_publication_evidence.append(missing)
                official_matches_without_publication_evidence.append(missing)
            fields = _official_match_mismatch_fields(
                match,
                game=draw_games_by_id[game_id],
                teams_by_id=draw_teams_by_id,
                tournament_id=clean_tournament_id,
                publication_projection=(
                    dict(publication_projection)
                    if isinstance(publication_projection, dict)
                    else None
                ),
            )
            if fields:
                mismatch = {
                    "match_id": str(match.get("id") or ""),
                    "tournament_game_id": game_id,
                    "fields": fields,
                }
                mismatched_matches.append(mismatch)
                mismatched_game_ids.add(game_id)
                official_match_payload_mismatches.append(mismatch)
        publication_counts = Counter(
            str(row.get("tournament_game_id") or "")
            for row in draw_matches
        )
        missing_publication_evidence_game_ids = {
            str(row.get("tournament_game_id") or "")
            for row in missing_publication_evidence
        }
        linked_ids = {game_id for game_id, count in publication_counts.items() if count > 0}
        duplicate_ids = sorted(game_id for game_id, count in publication_counts.items() if count > 1)
        published_ids = {
            game_id
            for game_id, count in publication_counts.items()
            if count == 1
            and game_id not in mismatched_game_ids
            and game_id not in missing_publication_evidence_game_ids
        }
        verified_published_game_ids.update(published_ids)
        if not matches_available:
            publication_state = "unavailable"
        elif duplicate_ids:
            publication_state = "duplicate"
        elif mismatched_matches:
            publication_state = "mismatch"
        elif draw_matches and missing_publication_evidence:
            publication_state = "evidence_unavailable"
        elif not linked_ids and not rating_publish_game_id_set:
            publication_state = "complete"
        elif not linked_ids:
            publication_state = "not_published"
        elif published_ids == rating_publish_game_id_set:
            publication_state = "complete"
        else:
            publication_state = "partial"
        publication_state_by_draw[draw_id] = publication_state

        team_summaries: list[dict[str, Any]] = []
        team_labels: dict[str, str] = {}
        for team in draw_teams:
            label, names = _team_label(team, player_names)
            team_id = str(team.get("id") or "")
            team_labels[team_id] = label
            team_summaries.append(
                {
                    "team_id": team_id,
                    "team_number": _safe_int(team.get("team_number")),
                    "name": label,
                    "player_names": names,
                }
            )
        rr_games = [
            row
            for row in competition_games
            if str(row.get("stage") or "").upper() == "ROUND_ROBIN"
        ]
        try:
            standings = compute_round_robin_standings(draw_teams, rr_games) if draw_teams else []
        except Exception:
            standings = []
        standings = [
            {
                **row,
                "team_name": team_labels.get(str(row.get("team_id") or ""), "Tournament team"),
            }
            for row in standings
        ]
        podium_summary = [
            {
                "placement": _safe_int(row.get("placement")),
                "team_id": str(row.get("team_id") or ""),
                "team_name": team_labels.get(str(row.get("team_id") or ""), "Tournament team"),
                "source": str(row.get("source") or "").upper(),
            }
            for row in draw_podium
        ]
        draw_operations = [
            {
                "operation_key": str(row.get("operation_key") or ""),
                "action": str(row.get("action") or ""),
                "status": str(row.get("status") or ""),
                "error_text": str(row.get("error_text") or "") or None,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in operations
            if str(row.get("entity_id") or "") == draw_id
        ][:20]
        states = {
            "live_operations": (
                "complete"
                if draw_counts["games"] and not draw_counts["open_games"]
                else "in_progress"
                if draw_counts["finalized_games"]
                else "not_started"
            ),
            "podium": "complete" if draw_counts["podium_complete"] else "blocked",
            "awards": "complete" if draw_counts["awards_complete"] else "blocked",
            "official_publish": (
                "complete" if publication_state == "complete" else "blocked"
            ),
        }
        draw_models.append(
            {
                "draw_id": draw_id,
                "name": draw_name,
                "status": str(draw.get("status") or "DRAFT").upper(),
                "protected": not _is_primary_draw(draw),
                "event_option_id": str(draw.get("event_option_id") or "") or None,
                "counts": {
                    "teams": len(draw_teams),
                    **draw_counts,
                    "rating_publish_eligible_games": len(rating_publish_game_ids),
                    "published_games": len(published_ids),
                    "unpublished_games": max(0, len(rating_publish_game_ids) - len(published_ids)),
                    "official_matches": sum(publication_counts.values()),
                    "duplicate_publications": len(duplicate_ids),
                    "duplicate_official_links": len(duplicate_ids),
                    "mismatched_official_matches": len(mismatched_matches),
                    "official_matches_without_publication_evidence": len(
                        missing_publication_evidence
                    ),
                },
                "teams": team_summaries,
                "standings": standings,
                "podium": podium_summary,
                "states": states,
                "operations": draw_operations,
                "review_evidence": review,
                "publication_evidence": {
                    "available": matches_available,
                    "state": publication_state,
                    "published_game_ids": sorted(published_ids),
                    "duplicate_game_ids": duplicate_ids,
                    "mismatched_matches": sorted(
                        mismatched_matches,
                        key=lambda row: (row["tournament_game_id"], row["match_id"]),
                    ),
                    "immutable_plan_available": immutable_plan_available,
                    "immutable_plan_operation_keys": list(
                        immutable_plan.get("operation_keys") or []
                    ),
                    "immutable_plan_errors": sorted(set(immutable_plan_errors)),
                    "matches_without_immutable_evidence": sorted(
                        missing_publication_evidence,
                        key=lambda row: (row["tournament_game_id"], row["match_id"]),
                    ),
                    "match_count": sum(publication_counts.values()),
                    "complete": publication_state == "complete",
                },
                "award_evidence": {
                    "available": awards_available,
                    "expected": expected_awards,
                    "expected_count": draw_counts["expected_awards"],
                    "verified_count": draw_counts["verified_awards"],
                    "active_count": draw_counts["active_awards"],
                    "complete": draw_counts["awards_complete"],
                },
                "readiness": {},
            }
        )
        total_expected_awards += int(draw_counts["expected_awards"])
        total_verified_awards += int(draw_counts["verified_awards"])
        total_finalized_games += int(draw_counts["finalized_games"])
        total_open_games += int(draw_counts["open_games"])
        total_tied_games += int(draw_counts["tied_games"])
        total_podium_entries += int(draw_counts["podium_entries"])
        total_podium_reviewed += int(bool(draw_counts["podium_reviewed"]))
        total_unexpected_awards += int(draw_counts["unexpected_awards"])

    publication_integrity_blockers: list[dict[str, Any]] = []
    for model in draw_models:
        draw_id = str(model["draw_id"])
        state = publication_state_by_draw.get(draw_id, "unavailable")
        if state == "partial":
            publication_integrity_blockers.append(
                _blocker(
                    "OFFICIAL_LINKS_PARTIAL",
                    f"{model['name']} has only part of its games linked to official Match Log rows.",
                    scope="draw",
                    draw_id=draw_id,
                )
            )
        elif state == "duplicate":
            publication_integrity_blockers.append(
                _blocker(
                    "OFFICIAL_LINKS_DUPLICATE",
                    f"{model['name']} has duplicate official Match Log links.",
                    scope="draw",
                    draw_id=draw_id,
                )
            )
        elif state == "mismatch":
            publication_integrity_blockers.append(
                _blocker(
                    "OFFICIAL_MATCH_PAYLOAD_MISMATCH",
                    f"{model['name']} has an official Match Log row whose immutable players, score, classification, or tournament context no longer matches its publication evidence.",
                    scope="draw",
                    draw_id=draw_id,
                    entity_type="match",
                    count=int(model["counts"]["mismatched_official_matches"]),
                )
            )
        elif state == "unavailable":
            publication_integrity_blockers.append(
                _blocker(
                    "OFFICIAL_LINK_EVIDENCE_UNAVAILABLE",
                    f"{model['name']} official Match Log links cannot be verified.",
                    scope="draw",
                    draw_id=draw_id,
                )
            )
        missing_publication_evidence_count = int(
            model["counts"].get("official_matches_without_publication_evidence") or 0
        )
        if matches_available and missing_publication_evidence_count:
            publication_integrity_blockers.append(
                _blocker(
                    "OFFICIAL_PUBLICATION_EVIDENCE_UNAVAILABLE",
                    f"{model['name']} already has {missing_publication_evidence_count} official Match Log link(s), but without one exact completed publication plan proving match_type, league, date, and week_tag.",
                    scope="draw",
                    draw_id=draw_id,
                    entity_type="match",
                    count=missing_publication_evidence_count,
                )
            )

    global_publish_base = _dedupe_blockers(
        [*all_core_blockers, *publication_integrity_blockers, *operation_blockers]
    )
    for model in draw_models:
        draw_id = str(model["draw_id"])
        blockers = list(global_publish_base)
        publication_state = publication_state_by_draw.get(draw_id, "unavailable")
        complete = publication_state == "complete"
        if complete:
            blockers.append(
                _blocker(
                    "DRAW_ALREADY_PUBLISHED",
                    f"{model['name']} already has exactly one official Match Log link per played game.",
                    scope="draw",
                    draw_id=draw_id,
                )
            )
        model["readiness"]["official_publish"] = _readiness(blockers, complete=complete)

    selected_model = next(
        (model for model in draw_models if str(model.get("draw_id") or "") == str(clean_selected_draw_id or "")),
        None,
    )
    if clean_selected_draw_id and selected_model:
        official_publish_readiness = dict(selected_model["readiness"]["official_publish"])
    else:
        candidates = [
            model["readiness"]["official_publish"]
            for model in draw_models
            if publication_state_by_draw.get(str(model["draw_id"])) == "not_published"
        ]
        selector_blockers = list(global_publish_base)
        if not draw_models:
            selector_blockers.append(
                _blocker(
                    "NO_ACTIVE_DRAWS",
                    "No tournament draw with teams or games is available for official publishing.",
                )
            )
        elif not candidates:
            selector_blockers.append(
                _blocker(
                    "NO_UNPUBLISHED_DRAWS",
                    "Every participating draw is already published or requires publication reconciliation.",
                )
            )
        official_publish_readiness = _readiness(selector_blockers)

    completion_blockers = list(global_publish_base)
    for model in draw_models:
        if publication_state_by_draw.get(str(model["draw_id"])) != "complete":
            remaining = max(
                0,
                int(model["counts"].get("rating_publish_eligible_games") or 0)
                - int(model["counts"]["published_games"]),
            )
            completion_blockers.append(
                _blocker(
                    "OFFICIAL_LINKS_INCOMPLETE",
                    f"{model['name']} needs exactly one official Match Log link for every played game.",
                    scope="draw",
                    draw_id=str(model["draw_id"]),
                    count=remaining,
                )
            )
    if not draw_models:
        completion_blockers.append(
            _blocker("NO_ACTIVE_DRAWS", "A tournament with no participating draws cannot be completed.")
        )
    tournament_status = str(tournament.get("status") or "").upper()
    completion_complete = (
        tournament_status in {"COMPLETED", "ARCHIVED"} and not completion_blockers
    )
    completion_readiness = _readiness(
        completion_blockers,
        complete=completion_complete,
    )
    archive_blockers = list(completion_blockers)
    if tournament_status not in {"COMPLETED", "ARCHIVED"}:
        archive_blockers.append(
            _blocker(
                "TOURNAMENT_NOT_COMPLETED",
                "Complete the tournament before moving it to the hidden archive.",
                entity_type="tournament",
                entity_id=clean_tournament_id,
            )
        )
    archive_complete = tournament_status == "ARCHIVED" and not completion_blockers
    archive_readiness = _readiness(archive_blockers, complete=archive_complete)
    for model in draw_models:
        model["readiness"]["completion"] = completion_readiness
        model["readiness"]["archive"] = archive_readiness

    total_games = sum(int(model["counts"]["games"]) for model in draw_models)
    total_rating_publish_games = sum(
        int(model["counts"].get("rating_publish_eligible_games") or 0)
        for model in draw_models
    )
    published_game_ids = set(verified_published_game_ids)
    duplicate_link_games = {
        game_id
        for game_id, rows in matches_by_game.items()
        if game_id in relevant_game_ids and len(rows) > 1
    }
    runtime = dict(runtime_capability or build_admin_tournament_ops_runtime_status())
    runtime.setdefault("official_publish_available", bool(runtime.get("official_publish_enabled")))
    terminal_writes_enabled = bool(runtime.get("tournament_mutations_enabled"))
    # Completion and the immutable closeout receipt are committed by the same
    # service-role-only database RPC.  Archive is a distinct visibility action
    # available only after the public terminal COMPLETED state exists.
    runtime["completion_atomic_commit_enabled"] = True
    runtime["completion_writes_enabled"] = terminal_writes_enabled
    runtime["completion_available"] = bool(
        terminal_writes_enabled and tournament_status not in {"COMPLETED", "ARCHIVED"}
    )
    runtime["archive_atomic_commit_enabled"] = True
    runtime["archive_writes_enabled"] = terminal_writes_enabled
    runtime["archive_available"] = bool(
        terminal_writes_enabled and tournament_status == "COMPLETED"
    )
    runtime["unarchive_available"] = bool(
        terminal_writes_enabled and tournament_status == "ARCHIVED"
    )

    if total_open_games:
        next_action = {
            "key": "continue_scoring",
            "label": "Continue scoring",
            "draw_id": str(clean_selected_draw_id or (draw_models[0]["draw_id"] if draw_models else "")) or None,
        }
    elif any(not bool(model["counts"]["podium_complete"]) for model in draw_models):
        next_action = {"key": "complete_podium", "label": "Complete podium", "draw_id": None}
    elif any(not bool(model["counts"]["podium_reviewed"]) for model in draw_models):
        next_action = {"key": "review_podium", "label": "Review podium", "draw_id": None}
    elif any(not bool(model["counts"]["awards_complete"]) for model in draw_models):
        next_action = {"key": "complete_awards", "label": "Complete awards", "draw_id": None}
    elif official_publish_readiness.get("ready"):
        next_action = {"key": "publish_official_matches", "label": "Publish official matches", "draw_id": clean_selected_draw_id}
    elif completion_readiness.get("ready") and runtime.get("completion_available"):
        next_action = {"key": "complete_tournament", "label": "Complete tournament", "draw_id": None}
    elif tournament_status == "COMPLETED" and runtime.get("archive_available"):
        next_action = {"key": "archive_tournament", "label": "Move to archive", "draw_id": None}
    elif tournament_status == "ARCHIVED" and runtime.get("unarchive_available"):
        next_action = {"key": "unarchive_tournament", "label": "Restore completed tournament", "draw_id": None}
    else:
        next_action = {"key": "resolve_blockers", "label": "Resolve closeout blockers", "draw_id": None}

    if tournament_status == "ARCHIVED":
        phase = "archived"
    elif tournament_status == "COMPLETED":
        phase = "completed"
    elif not draw_models:
        phase = "setup"
    elif total_open_games:
        phase = "live_in_progress" if total_finalized_games else "live_not_started"
    elif any(
        not bool(model["counts"][field])
        for model in draw_models
        for field in ("podium_complete", "podium_reviewed", "awards_complete")
    ):
        phase = "closeout_in_progress"
    elif completion_readiness.get("ready") and runtime.get("completion_available"):
        phase = "completion_ready"
    elif completion_readiness.get("ready"):
        phase = "completion_read_only"
    elif official_publish_readiness.get("ready"):
        phase = "publish_ready"
    else:
        phase = "publish_blocked"

    return {
        "ok": True,
        "contract": TOURNAMENT_LIFECYCLE_CONTRACT,
        "authority": TOURNAMENT_LIFECYCLE_AUTHORITY,
        "mode": "tournament_lifecycle",
        "scope": "draw" if clean_selected_draw_id else "tournament",
        "tournament": _tournament_payload(tournament),
        "phase": phase,
        "draw_id": clean_selected_draw_id,
        "selected_draw_id": clean_selected_draw_id,
        "counts": {
            "draws": len(draw_models),
            "active_team_parent_draws": len(active_team_parent_draws),
            "teams": sum(int(model["counts"]["teams"]) for model in draw_models),
            "orphan_teams": len(orphan_teams),
            "out_of_scope_teams": len(inactive_draw_teams),
            "games": total_games,
            "orphan_games": len(orphan_games),
            "out_of_scope_games": len(inactive_draw_games),
            "finalized_games": total_finalized_games,
            "open_games": total_open_games,
            "tied_games": total_tied_games,
            "podium_entries": total_podium_entries,
            "orphan_podium_entries": len(orphan_podium),
            "out_of_scope_podium_entries": len(inactive_draw_podium),
            "podiums_complete": sum(int(bool(model["counts"]["podium_complete"])) for model in draw_models),
            "podiums_reviewed": total_podium_reviewed,
            "expected_awards": total_expected_awards,
            "verified_awards": total_verified_awards,
            "unexpected_awards": total_unexpected_awards,
            "published_games": len(published_game_ids),
            "rating_publish_eligible_games": total_rating_publish_games,
            "unpublished_games": max(0, total_rating_publish_games - len(published_game_ids)),
            "official_matches": len(matches),
            "soft_deleted_official_matches": len(soft_deleted_matches),
            "mismatched_official_matches": len(official_match_payload_mismatches),
            "official_matches_without_publication_evidence": len(
                official_matches_without_publication_evidence
            ),
            "duplicate_publications": len(duplicate_link_games),
            "duplicate_official_links": len(duplicate_link_games),
            "active_operations": len(active_operations),
            "uncertain_operations": len(uncertain_operations),
            "recovery_required_operations": len(recovery_operations),
            "unsettled_match_exclusions": len(unsettled_match_exclusions),
            "unsettled_replay_jobs": len(unsettled_replay_jobs),
            "active_day_live_runs": len(active_day_live_runs),
        },
        "states": {
            "live_operations": (
                "complete"
                if total_games and not total_open_games
                else "in_progress"
                if total_finalized_games
                else "not_started"
            ),
            "official_publish": str(official_publish_readiness.get("state") or "blocked"),
            "completion": str(completion_readiness.get("state") or "blocked"),
            "archive": str(archive_readiness.get("state") or "blocked"),
        },
        "draws": draw_models,
        "domain_readiness": {
            "official_publish": official_publish_readiness,
            "completion": completion_readiness,
            "archive": archive_readiness,
        },
        "runtime_capability": runtime,
        "evidence": {
            "official_links_available": matches_available,
            "awards_available": awards_available,
            "operations_available": operations_available,
            "match_exclusions_available": match_exclusions_available,
            "replay_jobs_available": replay_jobs_available,
            "day_live_runs_available": day_live_runs_available,
            "players_available": players_available,
            "event_options_available": event_options_available,
            "active_team_parent_draw_ids": sorted(
                str(row.get("id") or "")
                for row in active_team_parent_draws
                if row.get("id")
            ),
            "orphan_team_ids": sorted(
                str(row.get("id") or "") for row in orphan_teams if row.get("id")
            ),
            "out_of_scope_team_ids": sorted(
                str(row.get("id") or "")
                for row in inactive_draw_teams
                if row.get("id")
            ),
            "orphan_game_ids": sorted(
                str(row.get("id") or "") for row in orphan_games if row.get("id")
            ),
            "out_of_scope_game_ids": sorted(
                str(row.get("id") or "")
                for row in inactive_draw_games
                if row.get("id")
            ),
            "orphan_podium_ids": sorted(
                str(row.get("id") or "") for row in orphan_podium if row.get("id")
            ),
            "out_of_scope_podium_ids": sorted(
                str(row.get("id") or "")
                for row in inactive_draw_podium
                if row.get("id")
            ),
            "soft_deleted_official_match_ids": sorted(
                str(row.get("id") or "")
                for row in soft_deleted_matches
                if row.get("id")
            ),
            "official_match_payload_mismatches": sorted(
                official_match_payload_mismatches,
                key=lambda row: (row["tournament_game_id"], row["match_id"]),
            ),
            "official_matches_without_publication_evidence": sorted(
                official_matches_without_publication_evidence,
                key=lambda row: (row["tournament_game_id"], row["match_id"]),
            ),
            "unsettled_match_exclusions": [
                {
                    "id": str(row.get("id") or ""),
                    "status": str(row.get("status") or ""),
                    "replay_job_id": str(row.get("replay_job_id") or "") or None,
                    "updated_at": row.get("updated_at"),
                }
                for row in unsettled_match_exclusions
            ],
            "unsettled_replay_jobs": [
                {
                    "id": str(row.get("id") or ""),
                    "status": str(row.get("status") or ""),
                    "updated_at": row.get("updated_at"),
                }
                for row in unsettled_replay_jobs
            ],
            "active_day_live_runs": [
                {
                    "id": str(row.get("id") or ""),
                    "state": str(row.get("state") or ""),
                    "updated_at": row.get("updated_at"),
                }
                for row in active_day_live_runs
            ],
            "active_operations": [
                {
                    "operation_key": str(row.get("operation_key") or ""),
                    "action": str(row.get("action") or ""),
                    "status": str(row.get("status") or ""),
                    "entity_id": str(row.get("entity_id") or ""),
                    "error_text": str(row.get("error_text") or "") or None,
                    "updated_at": row.get("updated_at"),
                }
                for row in active_operations
            ],
            "operations": [
                {
                    "operation_key": str(row.get("operation_key") or ""),
                    "action": str(row.get("action") or ""),
                    "status": str(row.get("status") or ""),
                    "entity_id": str(row.get("entity_id") or ""),
                    "error_text": str(row.get("error_text") or "") or None,
                    "updated_at": row.get("updated_at"),
                }
                for row in operations[:50]
            ],
            "recovery_required": len(recovery_operations),
        },
        "next_action": next_action,
        "warnings": list(dict.fromkeys(warnings)),
    }


def require_admin_tournament_official_publish_readiness(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    ignore_operation_key: str | None = None,
) -> dict[str, Any]:
    draw_rows, available, _ = _read_rows(
        supabase,
        "tournament_event_draws",
        filters=(("tournament_id", str(tournament_id)),),
    )
    target_draw = next(
        (row for row in draw_rows if str(row.get("id") or "") == str(draw_id)),
        None,
    )
    if not available:
        raise ValueError(
            "Tournament official publishing is blocked: draw evidence is unavailable."
        )
    if target_draw is None:
        raise ValueError("draw not found for this tournament")
    if _is_primary_draw(target_draw) and not _is_active_primary_draw(target_draw):
        raise ValueError(
            "Tournament official publishing is blocked: this draw is inactive."
        )
    protected_target = not _is_primary_draw(target_draw)
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        # Protected rating children are source artifacts, not operator draws.
        # Their specialized source validation runs in the publish service, but
        # they may never bypass the canonical tournament-wide closeout gate.
        selected_draw_id=None if protected_target else str(draw_id),
        ignore_operation_keys={str(ignore_operation_key)} if ignore_operation_key else None,
    )
    readiness = lifecycle["domain_readiness"]["official_publish"]
    completion_readiness = lifecycle["domain_readiness"]["completion"]
    ready = bool(readiness.get("ready")) or (
        protected_target and bool(completion_readiness.get("ready"))
    )
    if not ready:
        blocker_rows = list(readiness.get("blockers") or [])
        if protected_target:
            blocker_rows.extend(list(completion_readiness.get("blockers") or []))
        messages = [
            str(row.get("message") or "")
            for row in _dedupe_blockers(blocker_rows)
        ]
        raise ValueError(
            "Tournament official publishing is blocked: "
            + " ".join(message for message in messages if message)
        )
    if protected_target:
        lifecycle["protected_publish_target"] = {
            "draw_id": str(draw_id),
            "draw_kind": str(target_draw.get("draw_kind") or "STANDARD").upper(),
            "canonical_tournament_readiness": True,
        }
    return lifecycle


def require_admin_tournament_completion_readiness(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    ignore_operation_key: str | None = None,
) -> dict[str, Any]:
    lifecycle = build_admin_tournament_lifecycle(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
        ignore_operation_keys={str(ignore_operation_key)} if ignore_operation_key else None,
    )
    readiness = lifecycle["domain_readiness"]["completion"]
    if not bool(readiness.get("ready")):
        messages = [str(row.get("message") or "") for row in readiness.get("blockers") or []]
        raise ValueError(
            "Tournament completion is blocked: "
            + " ".join(message for message in messages if message)
        )
    return lifecycle


def require_admin_tournament_archive_readiness(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    ignore_operation_key: str | None = None,
) -> dict[str, Any]:
    """Compatibility alias for callers that still mean closeout readiness.

    The public terminal transition is now COMPLETED.  Moving an already
    completed tournament to ARCHIVED is a separate visibility action and is
    guarded by the terminal-status RPC.
    """

    return require_admin_tournament_completion_readiness(
        supabase,
        club_id=club_id,
        tournament_id=tournament_id,
        ignore_operation_key=ignore_operation_key,
    )


__all__ = [
    "build_admin_tournament_lifecycle",
    "require_admin_tournament_archive_readiness",
    "require_admin_tournament_completion_readiness",
    "require_admin_tournament_official_publish_readiness",
]

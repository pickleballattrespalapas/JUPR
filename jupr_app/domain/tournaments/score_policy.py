from __future__ import annotations

from typing import Any


SUPPORTED_SCORING_FORMATS = frozenset(
    {"GAME_TO_11", "GAME_TO_15", "GAME_TO_21", "BEST_2_OF_3"}
)
GAME_TARGETS = {
    "GAME_TO_11": 11,
    "GAME_TO_15": 15,
    "GAME_TO_21": 21,
}
BEST_OF_THREE_GAME_FORMAT = "GAME_TO_11"


def resolve_tournament_scoring_format(event: dict[str, Any] | None) -> str:
    """Resolve the authoritative per-division override before the event default."""

    row = event or {}
    resolved = str(
        row.get("scoring_override")
        or row.get("division_scoring")
        or row.get("scoring_default")
        or ""
    ).strip().upper()
    if resolved not in SUPPORTED_SCORING_FORMATS:
        raise ValueError(
            "Tournament scoring format is missing or unsupported. Configure GAME_TO_11, "
            "GAME_TO_15, GAME_TO_21, or BEST_2_OF_3 before recording a result."
        )
    return resolved


def review_tournament_score(
    score_a: Any,
    score_b: Any,
    *,
    scoring_format: str,
    unusual_score_acknowledged: bool = False,
) -> dict[str, Any]:
    """Classify a final score as ordinary, unusual, or impossible.

    GAME_TO_* uses a win-by-two target. A score above the target with a margin
    greater than two is structurally recordable (and often a fat-finger), so it
    requires an explicit acknowledgement rather than silently passing. Very
    long win-by-two deuce scores are also acknowledged. BEST_2_OF_3 is the
    derived parent result used by standings and bracket progression, so its
    only aggregate finals are 2-0 and 2-1. Played best-of-three results must
    additionally pass ``require_best_of_three_game_scores`` so every real game
    is preserved for ratings.
    """

    try:
        a = int(score_a)
        b = int(score_b)
    except (TypeError, ValueError) as exc:
        raise ValueError("Both scores must be whole numbers.") from exc
    if isinstance(score_a, float) and not score_a.is_integer():
        raise ValueError("Both scores must be whole numbers.")
    if isinstance(score_b, float) and not score_b.is_integer():
        raise ValueError("Both scores must be whole numbers.")

    format_code = str(scoring_format or "").strip().upper()
    if format_code not in SUPPORTED_SCORING_FORMATS:
        raise ValueError("Tournament scoring format is missing or unsupported.")

    reasons: list[str] = []
    impossible_reasons: list[str] = []
    if a < 0 or b < 0:
        impossible_reasons.append("Scores cannot be negative.")
    if a == b:
        impossible_reasons.append("Ties are not allowed in a final tournament result.")

    winner = max(a, b)
    loser = min(a, b)
    target: int | None = None
    win_by_two = format_code != "BEST_2_OF_3"
    if not impossible_reasons and format_code == "BEST_2_OF_3":
        if (winner, loser) not in {(2, 0), (2, 1)}:
            impossible_reasons.append(
                "BEST_2_OF_3 stores games won; the final must be 2-0 or 2-1."
            )
    elif not impossible_reasons:
        target = GAME_TARGETS[format_code]
        margin = winner - loser
        if winner < target:
            impossible_reasons.append(
                f"The winner must reach at least {target} points."
            )
        elif margin < 2:
            impossible_reasons.append("This format requires a two-point winning margin.")
        elif winner > target and margin != 2:
            reasons.append(
                f"The winning score is above {target} without a two-point deuce finish."
            )
        if winner > target + 20:
            reasons.append(
                f"The winning score is more than 20 points above the configured target of {target}."
            )

    status = "impossible" if impossible_reasons else "unusual" if reasons else "ordinary"
    acknowledged = bool(unusual_score_acknowledged)
    return {
        "status": status,
        "scoring_format": format_code,
        "target": target,
        "win_by_two": win_by_two,
        "score_a": a,
        "score_b": b,
        "reasons": impossible_reasons or reasons,
        "acknowledgement_required": status == "unusual",
        "acknowledged": acknowledged if status == "unusual" else False,
        "accepted": status == "ordinary" or (status == "unusual" and acknowledged),
    }


def require_tournament_score(
    score_a: Any,
    score_b: Any,
    *,
    scoring_format: str,
    unusual_score_acknowledged: bool = False,
) -> dict[str, Any]:
    review = review_tournament_score(
        score_a,
        score_b,
        scoring_format=scoring_format,
        unusual_score_acknowledged=unusual_score_acknowledged,
    )
    if review["status"] == "impossible":
        raise ValueError("Impossible tournament score: " + " ".join(review["reasons"]))
    if not review["accepted"]:
        raise ValueError(
            "Unusual tournament score requires explicit acknowledgement: "
            + " ".join(review["reasons"])
        )
    return review


def review_best_of_three_game_scores(
    game_scores: Any,
    *,
    unusual_score_acknowledged: bool = False,
) -> dict[str, Any]:
    """Validate the individual games in a best-two-of-three matchup.

    The parent tournament game keeps the derived 2-0 or 2-1 series result for
    standings and bracket progression.  Each row here is a real game to 11,
    win by two, and is preserved independently for rating publication.
    """

    if not isinstance(game_scores, list) or len(game_scores) not in {2, 3}:
        raise ValueError(
            "Best 2 of 3 requires two games for a 2-0 finish or three games for a 2-1 finish."
        )

    normalized: list[dict[str, Any]] = []
    wins_a = 0
    wins_b = 0
    unusual_reasons: list[str] = []
    for expected_number, raw in enumerate(game_scores, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Game {expected_number} must include both team scores.")
        try:
            game_number = int(raw.get("game_number"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Game {expected_number} must use its consecutive game number."
            ) from exc
        if game_number != expected_number:
            raise ValueError("Best-of-three game numbers must be consecutive starting at 1.")
        if wins_a == 2 or wins_b == 2:
            raise ValueError("A third game cannot be recorded after a team has won the first two games.")
        try:
            game_review = review_tournament_score(
                raw.get("score_a"),
                raw.get("score_b"),
                scoring_format=BEST_OF_THREE_GAME_FORMAT,
                unusual_score_acknowledged=unusual_score_acknowledged,
            )
        except ValueError as exc:
            raise ValueError(f"Game {game_number}: {exc}") from exc
        if game_review["status"] == "impossible":
            raise ValueError(
                f"Game {game_number}: " + " ".join(game_review["reasons"])
            )
        if game_review["status"] == "unusual":
            unusual_reasons.extend(
                f"Game {game_number}: {reason}"
                for reason in game_review.get("reasons") or []
            )
        score_a = int(game_review["score_a"])
        score_b = int(game_review["score_b"])
        if score_a > score_b:
            wins_a += 1
            winner_side = "A"
        else:
            wins_b += 1
            winner_side = "B"
        normalized.append(
            {
                "game_number": game_number,
                "score_a": score_a,
                "score_b": score_b,
                "winner_side": winner_side,
                "score_review": game_review,
            }
        )

    if max(wins_a, wins_b) != 2:
        raise ValueError(
            "Best 2 of 3 is incomplete. Enter the deciding third game after the teams split Games 1 and 2."
        )
    if len(normalized) != wins_a + wins_b:
        raise ValueError("Best-of-three game rows do not match the derived series result.")

    status = "unusual" if unusual_reasons else "ordinary"
    acknowledged = bool(unusual_score_acknowledged)
    return {
        "status": status,
        "scoring_format": "BEST_2_OF_3",
        "target": 2,
        "win_by_two": False,
        "individual_game_format": BEST_OF_THREE_GAME_FORMAT,
        "individual_game_target": GAME_TARGETS[BEST_OF_THREE_GAME_FORMAT],
        "individual_game_win_by_two": True,
        "score_a": wins_a,
        "score_b": wins_b,
        "game_scores": normalized,
        "reasons": unusual_reasons,
        "acknowledgement_required": status == "unusual",
        "acknowledged": acknowledged if status == "unusual" else False,
        "accepted": status == "ordinary" or (status == "unusual" and acknowledged),
    }


def require_best_of_three_game_scores(
    game_scores: Any,
    *,
    unusual_score_acknowledged: bool = False,
) -> dict[str, Any]:
    review = review_best_of_three_game_scores(
        game_scores,
        unusual_score_acknowledged=unusual_score_acknowledged,
    )
    if not review["accepted"]:
        raise ValueError(
            "Unusual tournament score requires explicit acknowledgement: "
            + " ".join(review["reasons"])
        )
    return review


__all__ = [
    "BEST_OF_THREE_GAME_FORMAT",
    "GAME_TARGETS",
    "SUPPORTED_SCORING_FORMATS",
    "require_best_of_three_game_scores",
    "require_tournament_score",
    "resolve_tournament_scoring_format",
    "review_best_of_three_game_scores",
    "review_tournament_score",
]

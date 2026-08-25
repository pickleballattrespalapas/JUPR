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
    long win-by-two deuce scores are also acknowledged. BEST_2_OF_3 stores
    games won, not individual game points, so its only finals are 2-0 and 2-1.
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


__all__ = [
    "GAME_TARGETS",
    "SUPPORTED_SCORING_FORMATS",
    "require_tournament_score",
    "resolve_tournament_scoring_format",
    "review_tournament_score",
]

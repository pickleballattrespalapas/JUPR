from __future__ import annotations

from typing import Any


ROUND_ROBIN_RANKING_POLICY: dict[str, Any] = {
    "description": "Rank active teams by wins. When teams are tied on wins, compare their record against the other tied teams first. If head-to-head does not fully separate them, resolve the remaining tie by point differential, then total points scored, then original team number. Retired teams remain visible after active teams and cannot advance.",
    "criteria": [
        "WINS",
        "HEAD_TO_HEAD",
        "POINT_DIFFERENTIAL",
        "POINTS_FOR",
        "TEAM_NUMBER",
    ],
    "retired_teams_eligible": False,
}


def round_robin_ranking_policy() -> dict[str, Any]:
    """Return a JSON-safe copy of the authoritative round-robin policy."""

    return {
        "description": str(ROUND_ROBIN_RANKING_POLICY["description"]),
        "criteria": list(ROUND_ROBIN_RANKING_POLICY["criteria"]),
        "retired_teams_eligible": bool(
            ROUND_ROBIN_RANKING_POLICY["retired_teams_eligible"]
        ),
    }


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _natural_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _tiebreak_count_label(count: int) -> str:
    words = {
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten",
    }
    return words.get(count, str(count))


def _tiebreak_outcome_sentence(
    outcome: str,
    *,
    criterion_label: str,
    groups_after: list[list[str]],
    team_name: Any,
) -> str:
    unresolved = [group for group in groups_after if len(group) > 1]
    normalized = _text(outcome).upper()
    if normalized == "RESOLVED":
        return f"{criterion_label} resolved the remaining tie."
    if normalized == "PARTIALLY_RESOLVED":
        remaining = "; ".join(
            _natural_list([team_name(team_id) for team_id in group])
            for group in unresolved
        )
        return (
            f"{criterion_label} separated some teams, but {remaining} "
            "remained tied."
        )
    return f"{criterion_label} did not separate these teams."


def _head_to_head_score_detail(
    team_ids: list[str],
    rr_games: list[dict[str, Any]],
    team_name: Any,
) -> str | None:
    if len(team_ids) != 2:
        return None
    matching: list[tuple[dict[str, Any], int, int]] = []
    expected_ids = set(team_ids)
    for game in rr_games:
        if {
            _text(game.get("team_a_id")),
            _text(game.get("team_b_id")),
        } != expected_ids:
            continue
        score_a = _safe_int(game.get("score_a"))
        score_b = _safe_int(game.get("score_b"))
        if score_a is None or score_b is None or score_a == score_b:
            continue
        matching.append((game, score_a, score_b))
    if len(matching) != 1:
        return None
    game, score_a, score_b = matching[0]
    if score_a > score_b:
        winner_id = _text(game.get("team_a_id"))
        loser_id = _text(game.get("team_b_id"))
        winner_score, loser_score = score_a, score_b
    else:
        winner_id = _text(game.get("team_b_id"))
        loser_id = _text(game.get("team_a_id"))
        winner_score, loser_score = score_b, score_a
    result_type = _text(game.get("result_type") or "PLAYED").upper()
    result_note = (
        f" ({result_type.replace('_', ' ').lower()})"
        if result_type != "PLAYED"
        else ""
    )
    return (
        f"{team_name(winner_id)} defeated {team_name(loser_id)} "
        f"{winner_score}\u2013{loser_score}{result_note} in their head-to-head matchup."
    )


def build_round_robin_tiebreak_explanations(
    tiebreaks: list[dict[str, Any]],
    standings: list[dict[str, Any]],
    rr_games: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format the authoritative audit without exposing internal team identifiers."""

    standings_by_id = {
        _text(row.get("team_id")): row
        for row in standings
        if _text(row.get("team_id"))
    }

    def team_name(team_id: str) -> str:
        row = standings_by_id.get(_text(team_id)) or {}
        name = _text(row.get("team_name"))
        if name:
            return name
        team_number = _safe_int(row.get("team_number"))
        return (
            f"Team {team_number}"
            if team_number is not None
            else "Team name unavailable"
        )

    explanations: list[dict[str, Any]] = []
    criterion_labels = {
        "HEAD_TO_HEAD": "Head-to-head",
        "POINT_DIFFERENTIAL": "Point differential",
        "POINTS_FOR": "Total points scored",
        "TEAM_NUMBER": "Original team number",
    }
    for audit in tiebreaks:
        team_ids = [
            _text(team_id)
            for team_id in list(audit.get("team_ids") or [])
            if _text(team_id)
        ]
        final_team_ids = [
            _text(team_id)
            for team_id in list(audit.get("final_team_ids") or [])
            if _text(team_id)
        ]
        if len(team_ids) < 2 or len(final_team_ids) != len(team_ids):
            continue
        wins = _safe_int(audit.get("wins"), 0) or 0
        losses = {
            _safe_int((standings_by_id.get(team_id) or {}).get("losses"), 0) or 0
            for team_id in team_ids
        }
        record = (
            f"{wins}\u2013{next(iter(losses))}"
            if len(losses) == 1
            else f"{wins} wins"
        )
        title = f"{_tiebreak_count_label(len(team_ids))}-way tie at {record}"
        steps: list[dict[str, Any]] = []
        head_to_head_incomplete = False
        for raw_step in list(audit.get("steps") or []):
            criterion = _text(raw_step.get("criterion")).upper()
            if criterion not in criterion_labels:
                continue
            outcome = _text(raw_step.get("outcome")).upper()
            groups_before = [
                [_text(team_id) for team_id in list(group or []) if _text(team_id)]
                for group in list(raw_step.get("groups_before") or [])
            ]
            groups_after = [
                [_text(team_id) for team_id in list(group or []) if _text(team_id)]
                for group in list(raw_step.get("groups_after") or [])
            ]
            values_by_id = {
                _text(item.get("team_id")): item
                for item in list(raw_step.get("team_values") or [])
                if _text(item.get("team_id"))
            }
            criterion_label = criterion_labels[criterion]
            outcome_sentence = _tiebreak_outcome_sentence(
                outcome,
                criterion_label=criterion_label,
                groups_after=groups_after,
                team_name=team_name,
            )
            if criterion == "HEAD_TO_HEAD":
                head_to_head_complete = raw_step.get("complete") is not False
                head_to_head_incomplete = not head_to_head_complete
                if not head_to_head_complete:
                    records = []
                    for group in groups_before:
                        for team_id in group:
                            value = values_by_id.get(team_id) or {}
                            records.append(
                                f"{team_name(team_id)} "
                                f"{_safe_int(value.get('wins'), 0) or 0}\u2013"
                                f"{_safe_int(value.get('losses'), 0) or 0}"
                            )
                    missing_pair_labels = [
                        f"{team_name(pair[0])} vs {team_name(pair[1])}"
                        for pair in list(raw_step.get("missing_pairs") or [])
                        if isinstance(pair, (list, tuple)) and len(pair) == 2
                    ]
                    available_detail = (
                        f"Available head-to-head records: {'; '.join(records)}. "
                        if list(raw_step.get("matchups") or [])
                        else "No head-to-head result was available. "
                    )
                    missing_detail = (
                        "The complete comparison was unavailable because "
                        f"{'this matchup' if len(missing_pair_labels) == 1 else 'these matchups'} "
                        "had no scored result: "
                        f"{_natural_list(missing_pair_labels)}. "
                        if missing_pair_labels
                        else "The complete comparison was unavailable. "
                    )
                    detail = (
                        f"{available_detail}{missing_detail}"
                        "Head-to-head was not applied."
                    )
                else:
                    score_detail = _head_to_head_score_detail(
                        groups_before[0] if len(groups_before) == 1 else [],
                        rr_games,
                        team_name,
                    )
                    if score_detail:
                        detail = f"{score_detail} {outcome_sentence}"
                    else:
                        records = []
                        for group in groups_before:
                            for team_id in group:
                                value = values_by_id.get(team_id) or {}
                                records.append(
                                    f"{team_name(team_id)} "
                                    f"{_safe_int(value.get('wins'), 0) or 0}\u2013"
                                    f"{_safe_int(value.get('losses'), 0) or 0}"
                                )
                        detail = (
                            f"Head-to-head mini-table: {'; '.join(records)}. "
                            f"{outcome_sentence}"
                        )
            else:
                descending = criterion != "TEAM_NUMBER"
                value_groups: list[str] = []
                for group in groups_before:
                    ordered_group = sorted(
                        group,
                        key=lambda team_id: _safe_int(
                            (values_by_id.get(team_id) or {}).get("value"), 0
                        )
                        or 0,
                        reverse=descending,
                    )
                    formatted: list[str] = []
                    for team_id in ordered_group:
                        value = _safe_int(
                            (values_by_id.get(team_id) or {}).get("value"), 0
                        ) or 0
                        if criterion == "POINT_DIFFERENTIAL":
                            display_value = f"{value:+d}"
                        elif criterion == "TEAM_NUMBER":
                            display_value = f"Team {value}"
                        else:
                            display_value = str(value)
                        formatted.append(f"{team_name(team_id)} {display_value}")
                    value_groups.append("; ".join(formatted))
                detail = (
                    f"{criterion_label} for the remaining tied teams: "
                    f"{' | '.join(value_groups)}. {outcome_sentence}"
                )
            steps.append(
                {
                    "criterion": criterion,
                    "outcome": outcome,
                    "detail": detail,
                }
            )
        if not steps:
            continue
        final_order = " \u2192 ".join(
            team_name(team_id) for team_id in final_team_ids
        )
        final_criterion = steps[-1]["criterion"]
        if final_criterion == "HEAD_TO_HEAD":
            summary = f"Head-to-head resolved the tie. Final order: {final_order}."
        elif head_to_head_incomplete:
            summary = (
                "A complete head-to-head comparison was unavailable, so "
                f"{criterion_labels[final_criterion].lower()} completed the order: "
                f"{final_order}."
            )
        elif final_criterion == "TEAM_NUMBER":
            summary = (
                "The competitive tie-breaks remained level, so original team "
                f"number set the deterministic final order: {final_order}."
            )
        else:
            summary = (
                "Head-to-head did not fully separate these teams. "
                f"{criterion_labels[final_criterion]} completed the order: "
                f"{final_order}."
            )
        explanations.append(
            {
                "title": title,
                "summary": summary,
                "steps": steps,
            }
        )
    return explanations


__all__ = [
    "ROUND_ROBIN_RANKING_POLICY",
    "build_round_robin_tiebreak_explanations",
    "round_robin_ranking_policy",
]

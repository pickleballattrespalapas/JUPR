from __future__ import annotations

from typing import Any, Iterable, Mapping

AGE_POLICY_MODES = {
    "ALL_AGES",
    "FIXED_AGE_BRACKET",
    "SPLIT_AGE",
    "AUTO_AGE_SPLIT",
}
TEAM_AGE_RULES = {"YOUNGER", "OLDER", "AVERAGE", "BOTH_QUALIFY"}
AGE_MERGE_STRATEGIES = {"CLOSEST", "UP", "DOWN"}


def _clean(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def normalize_age_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize an event or division age policy.

    The returned bracket list is authoritative for preview assignment. Gaps are
    allowed intentionally so the preview can identify entries that need manual
    resolution; overlaps are never allowed because they make assignment
    ambiguous.
    """

    raw = _mapping(policy)
    mode = _clean(raw.get("mode"), limit=40).upper() or "ALL_AGES"
    if mode not in AGE_POLICY_MODES:
        raise ValueError("Invalid age policy mode.")

    team_age_rule = _clean(raw.get("team_age_rule"), limit=40).upper() or "YOUNGER"
    if team_age_rule not in TEAM_AGE_RULES:
        raise ValueError("Invalid team age rule.")

    merge_strategy = _clean(raw.get("merge_strategy"), limit=40).upper() or "CLOSEST"
    if merge_strategy not in AGE_MERGE_STRATEGIES:
        raise ValueError("Invalid age merge strategy.")

    raw_minimum = raw.get("min_teams_per_age_group")
    try:
        minimum = int(1 if raw_minimum in (None, "") else raw_minimum)
    except (TypeError, ValueError) as exc:
        raise ValueError("Minimum entries per age group must be a whole number.") from exc
    if minimum < 1:
        raise ValueError("Minimum entries per age group must be at least 1.")

    brackets: list[dict[str, Any]] = []
    if mode == "ALL_AGES":
        brackets = [
            {
                "id": "all-ages",
                "label": "All ages",
                "min_age": None,
                "max_age": None,
            }
        ]
    elif mode == "FIXED_AGE_BRACKET":
        minimum_age = optional_number(raw.get("min_age"))
        maximum_age = optional_number(raw.get("max_age"))
        if minimum_age is not None and maximum_age is not None and maximum_age < minimum_age:
            raise ValueError("Fixed age bracket has a maximum below its minimum.")
        brackets = [
            {
                "id": "fixed-age",
                "label": _clean(raw.get("label"), limit=80) or "Fixed age bracket",
                "min_age": minimum_age,
                "max_age": maximum_age,
            }
        ]
    elif mode == "SPLIT_AGE":
        threshold = optional_number(raw.get("split_age_threshold"))
        if threshold is None or threshold < 1:
            raise ValueError("Split-age partners threshold must be at least 1.")
        threshold_int = int(threshold)
        brackets = [
            {
                "id": f"split-age-{threshold_int}",
                "label": f"One under {threshold_int} / one {threshold_int}+",
                "min_age": None,
                "max_age": None,
            }
        ]
    else:
        raw_brackets = _mapping_rows(raw.get("brackets"))
        if len(raw_brackets) < 2:
            raise ValueError("Auto age split requires at least two candidate brackets.")

        labels: set[str] = set()
        previous_minimum: float | None = None
        previous_maximum: float | None = None
        for index, row in enumerate(raw_brackets, start=1):
            label = _clean(row.get("label"), limit=80)
            if not label:
                raise ValueError(f"Age bracket {index} needs a label.")
            normalized_label = label.lower()
            if normalized_label in labels:
                raise ValueError("Age bracket labels must be unique.")
            labels.add(normalized_label)

            minimum_age = optional_number(row.get("min_age"))
            maximum_age = optional_number(row.get("max_age"))
            if minimum_age is not None and maximum_age is not None and maximum_age < minimum_age:
                raise ValueError(f"Age bracket '{label}' has a maximum below its minimum.")
            if index > 1 and minimum_age is None:
                raise ValueError("Only the first auto age bracket may omit a minimum age.")
            if index < len(raw_brackets) and maximum_age is None:
                raise ValueError("Only the final auto age bracket may omit a maximum age.")
            if previous_minimum is not None and minimum_age is not None and minimum_age < previous_minimum:
                raise ValueError("Auto age brackets must be ordered by minimum age.")
            if previous_maximum is not None and minimum_age is not None and minimum_age <= previous_maximum:
                raise ValueError("Auto age brackets must be ordered and may not overlap.")

            previous_minimum = minimum_age if minimum_age is not None else previous_minimum
            previous_maximum = maximum_age
            brackets.append(
                {
                    "id": _clean(row.get("id"), limit=80) or f"age-{index}",
                    "label": label,
                    "min_age": minimum_age,
                    "max_age": maximum_age,
                }
            )

    return {
        "mode": mode,
        "label": _clean(raw.get("label"), limit=80),
        "split_age_threshold": int(optional_number(raw.get("split_age_threshold")) or 0) or None,
        "min_teams_per_age_group": minimum,
        "team_age_rule": team_age_rule,
        "merge_strategy": merge_strategy,
        "brackets": brackets,
    }


def age_in_bracket(age: float, bracket: Mapping[str, Any]) -> bool:
    minimum = optional_number(bracket.get("min_age"))
    maximum = optional_number(bracket.get("max_age"))
    return (minimum is None or age >= minimum) and (maximum is None or age <= maximum)


def effective_team_age(
    player_age: float | None,
    partner_age: float | None,
    rule: str,
) -> float | None:
    ages = [age for age in (player_age, partner_age) if age is not None]
    if not ages:
        return None
    if len(ages) == 1:
        return ages[0]
    normalized_rule = _clean(rule, limit=40).upper() or "YOUNGER"
    if normalized_rule == "OLDER":
        return max(ages)
    if normalized_rule == "AVERAGE":
        return round(sum(ages) / len(ages), 1)
    return min(ages)


def registration_display_name(registration: Mapping[str, Any]) -> str:
    registration_id = _clean(registration.get("id"), limit=120)
    return (
        _clean(registration.get("display_name"), limit=180)
        or " ".join(
            part
            for part in (
                _clean(registration.get("first_name"), limit=80),
                _clean(registration.get("last_name"), limit=80),
            )
            if part
        )
        or _clean(registration.get("email"), limit=180)
        or registration_id
    )


def _recommended_merge_target(
    brackets: list[dict[str, Any]],
    index: int,
    strategy: str,
) -> dict[str, Any] | None:
    if strategy == "UP" and index + 1 < len(brackets):
        return brackets[index + 1]
    if strategy == "DOWN" and index > 0:
        return brackets[index - 1]
    candidates: list[dict[str, Any]] = []
    if index > 0:
        candidates.append(brackets[index - 1])
    if index + 1 < len(brackets):
        candidates.append(brackets[index + 1])
    if not candidates:
        return None
    # Prefer the adjacent group with more entries, then the earlier group for
    # deterministic output when counts are equal.
    return max(
        candidates,
        key=lambda row: (
            int(row.get("count") or 0),
            -brackets.index(row),
        ),
    )


def build_age_split_preview(
    *,
    policy: Mapping[str, Any] | None,
    registrations: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    selections: Iterable[Mapping[str, Any]],
    participant_type: str,
) -> dict[str, Any]:
    """Build a deterministic, zero-write age split preview from registration rows."""

    normalized_policy = normalize_age_policy(policy)
    if isinstance(registrations, Mapping):
        registration_by_id = {
            str(key): dict(value)
            for key, value in registrations.items()
            if isinstance(value, Mapping)
        }
    else:
        registration_by_id = {
            str(row.get("id") or "").strip(): dict(row)
            for row in registrations
            if isinstance(row, Mapping) and str(row.get("id") or "").strip()
        }

    participant = _clean(participant_type, limit=40).upper() or "GENDER_DOUBLES"
    team_event = participant != "SINGLES"
    split_age_mode = normalized_policy["mode"] == "SPLIT_AGE"
    if split_age_mode and not team_event:
        raise ValueError("Split-age partners is available only for doubles and team events.")
    split_threshold = optional_number(normalized_policy.get("split_age_threshold"))
    preview_brackets = [
        {
            **bracket,
            "count": 0,
            "viable": False,
            "entries": [],
        }
        for bracket in normalized_policy["brackets"]
    ]
    unassigned: list[dict[str, Any]] = []
    processed = 0

    for raw_selection in selections:
        if not isinstance(raw_selection, Mapping):
            continue
        selection = dict(raw_selection)
        registration_id = str(selection.get("registration_id") or "").strip()
        registration = registration_by_id.get(registration_id)
        if registration is None:
            continue
        processed += 1

        player_age = optional_number(registration.get("age"))
        partner_age = optional_number(selection.get("partner_age"))
        effective_age = effective_team_age(
            player_age,
            partner_age if team_event else None,
            str(normalized_policy["team_age_rule"]),
        )
        entry = {
            "registration_id": registration_id,
            "selection_id": str(selection.get("id") or "").strip() or None,
            "display_name": registration_display_name(registration),
            "age": player_age,
            "partner_age": partner_age,
            "effective_age": effective_age,
        }

        assigned = False
        if split_age_mode:
            matches = (
                split_threshold is not None
                and player_age is not None
                and partner_age is not None
                and (
                    (player_age < split_threshold <= partner_age)
                    or (partner_age < split_threshold <= player_age)
                )
            )
            if matches:
                preview_brackets[0]["entries"].append(entry)
                preview_brackets[0]["count"] += 1
                assigned = True
            else:
                entry["assignment_issue"] = (
                    f"Team must include one player under {int(split_threshold or 0)} "
                    f"and one player {int(split_threshold or 0)}+."
                )
        else:
            for bracket in preview_brackets:
                if normalized_policy["team_age_rule"] == "BOTH_QUALIFY" and team_event:
                    matches = (
                        player_age is not None
                        and partner_age is not None
                        and age_in_bracket(player_age, bracket)
                        and age_in_bracket(partner_age, bracket)
                    )
                else:
                    matches = effective_age is not None and age_in_bracket(effective_age, bracket)
                if matches:
                    bracket["entries"].append(entry)
                    bracket["count"] += 1
                    assigned = True
                    break
        if not assigned:
            unassigned.append(entry)

    # Split-age partners is a team-composition rule, not an automatic bracket
    # split. Any qualifying team makes the single preview group viable; the
    # minimum-per-bracket setting applies only to actual age-bracket modes.
    minimum = 1 if split_age_mode else int(normalized_policy["min_teams_per_age_group"])
    for bracket in preview_brackets:
        bracket["viable"] = int(bracket["count"]) >= minimum

    recommendations: list[str] = []
    strategy = str(normalized_policy["merge_strategy"])
    for index, bracket in enumerate(preview_brackets):
        if bracket["viable"] or bracket["count"] == 0:
            continue
        target = _recommended_merge_target(preview_brackets, index, strategy)
        if target is not None:
            recommendations.append(
                f"Merge {bracket['label']} ({bracket['count']}) into "
                f"{target['label']} ({target['count']}) before accepting the split."
            )
        else:
            recommendations.append(
                f"{bracket['label']} has {bracket['count']} entries, below the minimum of {minimum}."
            )
    if unassigned:
        if split_age_mode:
            recommendations.append(
                f"Resolve {len(unassigned)} team entr{'y' if len(unassigned) == 1 else 'ies'} "
                "that do not meet the one-under / one-over split-age rule before accepting the event setup."
            )
        else:
            recommendations.append(
                f"Resolve ages for {len(unassigned)} unassigned "
                f"entr{'y' if len(unassigned) == 1 else 'ies'} before accepting the split."
            )

    return {
        "policy": normalized_policy,
        "total_entries": processed,
        "brackets": preview_brackets,
        "recommendations": recommendations,
        "unassigned_entries": unassigned,
    }

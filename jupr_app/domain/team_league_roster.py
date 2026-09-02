from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


TEAM_LEAGUE_TEAM_SIZES = frozenset({2, 3, 4})
TEAM_LEAGUE_CATEGORIES = frozenset({"open", "mens", "womens", "mixed"})
TEAM_LEAGUE_MEMBER_ROLES = frozenset({"captain", "primary", "alternate"})
TEAM_LEAGUE_ACTIVE_MEMBER_STATUSES = frozenset({"invited", "active"})
TEAM_LEAGUE_POOL_STATUSES = frozenset({"available", "unavailable", "withdrawn"})


def normalize_team_category(value: Any) -> str:
    clean = str(value or "").strip().lower().replace("’", "'")
    aliases = {
        "men": "mens",
        "men's": "mens",
        "male": "mens",
        "women": "womens",
        "women's": "womens",
        "female": "womens",
        "coed": "mixed",
    }
    return aliases.get(clean, clean or "open")


def normalize_player_gender(value: Any) -> str | None:
    clean = (
        str(value or "")
        .strip()
        .lower()
        .replace("’", "'")
        .replace("_", " ")
    )
    aliases = {
        "m": "male",
        "man": "male",
        "men": "male",
        "male": "male",
        "men's": "male",
        "mens": "male",
        "f": "female",
        "w": "female",
        "woman": "female",
        "women": "female",
        "female": "female",
        "women's": "female",
        "womens": "female",
    }
    return aliases.get(clean)


def default_mixed_counts(team_size: int) -> tuple[int, int]:
    """Return a deterministic balanced primary-roster split for sizes 2-4."""

    if team_size not in TEAM_LEAGUE_TEAM_SIZES:
        raise ValueError("Team size must be 2, 3, or 4 players.")
    return team_size // 2, team_size - (team_size // 2)


def normalize_roster_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    try:
        raw_team_size = settings.get("team_size")
        team_size = int(raw_team_size if raw_team_size not in (None, "") else 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Team size must be 2, 3, or 4 players.") from exc
    if team_size not in TEAM_LEAGUE_TEAM_SIZES:
        raise ValueError("Team size must be 2, 3, or 4 players.")

    try:
        raw_max_alternates = settings.get("max_alternates")
        max_alternates = int(
            raw_max_alternates if raw_max_alternates not in (None, "") else 0
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Maximum alternates must be a whole number from 0 to 4.") from exc
    if max_alternates < 0 or max_alternates > 4:
        raise ValueError("Maximum alternates must be a whole number from 0 to 4.")

    category = normalize_team_category(settings.get("team_category"))
    if category not in TEAM_LEAGUE_CATEGORIES:
        raise ValueError("Choose Open, Men's, Women's, or Mixed team eligibility.")

    default_men, default_women = default_mixed_counts(team_size)
    try:
        required_men = int(
            settings.get("mixed_required_men")
            if settings.get("mixed_required_men") not in (None, "")
            else default_men
        )
        required_women = int(
            settings.get("mixed_required_women")
            if settings.get("mixed_required_women") not in (None, "")
            else default_women
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Mixed roster counts must be whole numbers.") from exc
    composition_valid = (
        required_men >= 1
        and required_women >= 1
        and required_men + required_women == team_size
    )
    if category == "mixed" and not composition_valid:
        raise ValueError(
            "Mixed roster counts must each be at least one and total the team size."
        )
    if not composition_valid:
        required_men, required_women = default_men, default_women

    return {
        "team_size": team_size,
        "team_category": category,
        "max_alternates": max_alternates,
        "substitute_pool_enabled": bool(settings.get("substitute_pool_enabled")),
        "mixed_required_men": required_men,
        "mixed_required_women": required_women,
    }


def _player_label(row: Mapping[str, Any]) -> str:
    name = str(row.get("name") or row.get("player_name") or "").strip()
    return name or f"Player {row.get('id') or row.get('player_id') or ''}".strip()


def validate_team_members(
    *,
    settings: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    require_complete: bool,
) -> None:
    """Validate assigned primary players and alternates against league policy."""

    policy = normalize_roster_settings(settings)
    current = [
        row
        for row in members
        if str(row.get("status") or "active").lower()
        in TEAM_LEAGUE_ACTIVE_MEMBER_STATUSES
    ]
    primary = [
        row
        for row in current
        if str(row.get("role") or "primary").lower() in {"captain", "primary"}
    ]
    alternates = [
        row for row in current if str(row.get("role") or "").lower() == "alternate"
    ]
    player_ids = [int(row.get("player_id") or row.get("id")) for row in current]
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("A player can appear only once on a team roster.")
    if len(primary) > policy["team_size"]:
        raise ValueError("This team has more primary players than its configured size.")
    if require_complete and len(primary) != policy["team_size"]:
        raise ValueError(
            f"A confirmed team needs exactly {policy['team_size']} active primary players."
        )
    if len(alternates) > policy["max_alternates"]:
        raise ValueError("This team has more alternates than the league allows.")

    category = policy["team_category"]
    if category == "open":
        return
    genders = [normalize_player_gender(row.get("gender")) for row in current]
    unresolved = [
        _player_label(row)
        for row, gender in zip(current, genders)
        if gender is None
    ]
    if unresolved:
        raise ValueError(
            "Team eligibility cannot verify the gender for "
            + ", ".join(unresolved)
            + ". Update the player profile gender and try again."
        )
    if category == "mens" and any(gender != "male" for gender in genders):
        raise ValueError("Men's team eligibility requires players marked Male.")
    if category == "womens" and any(gender != "female" for gender in genders):
        raise ValueError("Women's team eligibility requires players marked Female.")
    if category == "mixed":
        primary_genders = [
            normalize_player_gender(row.get("gender")) for row in primary
        ]
        counts = Counter(primary_genders)
        composition_impossible = (
            counts["male"] > policy["mixed_required_men"]
            or counts["female"] > policy["mixed_required_women"]
        )
        composition_incomplete = require_complete and (
            counts["male"] != policy["mixed_required_men"]
            or counts["female"] != policy["mixed_required_women"]
        )
        if composition_impossible or composition_incomplete:
            raise ValueError(
                "Mixed primary roster eligibility requires "
                f"{policy['mixed_required_men']} men and "
                f"{policy['mixed_required_women']} women."
            )


def validate_playing_lineup(
    *, category: Any, player_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Apply team category policy to the two players who actually take court."""

    if len(player_rows) != 2:
        raise ValueError("Each side needs exactly two players.")
    clean_category = normalize_team_category(category)
    if clean_category == "open":
        return
    genders = [normalize_player_gender(row.get("gender")) for row in player_rows]
    unresolved = [
        _player_label(row)
        for row, gender in zip(player_rows, genders)
        if gender is None
    ]
    if unresolved:
        raise ValueError(
            "Lineup eligibility cannot verify the gender for "
            + ", ".join(unresolved)
            + "."
        )
    eligible = (
        clean_category == "mens" and genders == ["male", "male"]
    ) or (
        clean_category == "womens" and genders == ["female", "female"]
    ) or (
        clean_category == "mixed" and set(genders) == {"male", "female"}
    )
    if not eligible:
        requirement = {
            "mens": "two men",
            "womens": "two women",
            "mixed": "one man and one woman",
        }[clean_category]
        raise ValueError(f"This playing lineup requires {requirement}.")

from __future__ import annotations

from uuid import uuid4

from jupr_app.data.retry import sb_retry


def _next_power_of_two(value: int) -> int:
    size = 1
    while size < value:
        size *= 2
    return size


def generate_single_elim(supabase, division_id: str, club_id: str) -> dict[str, int]:
    clean_division_id = str(division_id or "").strip()
    clean_club_id = str(club_id or "").strip()
    if not clean_division_id or not clean_club_id:
        raise ValueError("Division ID and club ID are required.")

    entries_resp = sb_retry(
        lambda: (
            supabase.table("division_entries")
            .select("team_id,seed,created_at")
            .eq("club_id", clean_club_id)
            .eq("division_id", clean_division_id)
            .order("seed", desc=False, nullsfirst=False)
            .order("created_at", desc=False)
            .execute()
        )
    )
    entries = entries_resp.data or []

    if len(entries) < 2:
        raise ValueError("At least 2 teams are required to generate a bracket.")

    existing_resp = sb_retry(
        lambda: (
            supabase.table("division_matches")
            .select("id")
            .eq("club_id", clean_club_id)
            .eq("division_id", clean_division_id)
            .limit(1)
            .execute()
        )
    )
    if existing_resp.data:
        raise ValueError("Bracket already generated for this division.")

    bracket_size = _next_power_of_two(len(entries))
    slots: list[str | None] = [None] * bracket_size
    for idx, entry in enumerate(entries):
        team_id = entry.get("team_id")
        if team_id:
            slots[idx] = str(team_id)

    matchup_rows: list[dict] = []
    bracket_position = 1
    half = bracket_size // 2
    for idx in range(half):
        team_a_id = slots[idx]
        team_b_id = slots[bracket_size - 1 - idx]
        if not team_a_id or not team_b_id:
            continue
        matchup_rows.append(
            {
                "id": str(uuid4()),
                "club_id": clean_club_id,
                "division_id": clean_division_id,
                "round_number": 1,
                "bracket_position": bracket_position,
                "team_a_id": team_a_id,
                "team_b_id": team_b_id,
                "status": "scheduled",
            }
        )
        bracket_position += 1

    if not matchup_rows:
        raise ValueError("Unable to create round 1 matchups from current entries.")

    sb_retry(lambda: supabase.table("division_matches").insert(matchup_rows).execute())
    sb_retry(
        lambda: (
            supabase.table("tournament_divisions")
            .update({"status": "active"})
            .eq("club_id", clean_club_id)
            .eq("id", clean_division_id)
            .execute()
        )
    )

    return {
        "entry_count": len(entries),
        "bracket_size": bracket_size,
        "match_count": len(matchup_rows),
    }

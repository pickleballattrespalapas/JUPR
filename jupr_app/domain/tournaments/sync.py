from __future__ import annotations

from typing import Any


def validate_podium_placements(placements: list[dict[str, Any]], *, max_placements: int = 3) -> None:
    seen_team_ids: set[str] = set()
    for placement in placements:
        place = int(placement.get("placement", 0) or 0)
        if place < 1 or place > max_placements:
            raise ValueError("Podium placement must be between 1 and 3.")
        team_id = placement.get("team_id")
        if not team_id:
            raise ValueError("Podium placement requires a team.")
        if team_id in seen_team_ids:
            raise ValueError("Podium placements must use distinct teams.")
        seen_team_ids.add(team_id)


def build_podium_payload(tournament_id: str, placements: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    ordered = sorted(placements, key=lambda row: int(row.get("placement", 0) or 0))
    validate_podium_placements(ordered, max_placements=3)
    return [
        {
            "tournament_id": tournament_id,
            "placement": int(row["placement"]),
            "team_id": row["team_id"],
            "source": source,
        }
        for row in ordered
    ]

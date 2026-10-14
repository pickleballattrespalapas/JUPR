"""Resolve club ratings on the server; never accept client supplied ratings."""
from __future__ import annotations

import math
from typing import Any

from jupr_app.domain.moneyball_expectations import freeze_moneyball, is_moneyball


def prepare_moneyball(supabase: Any, *, club_id: str, event: dict[str, Any]) -> dict[str, Any]:
    if not is_moneyball(event):
        return event
    participants = event["participants"]
    ids = [p.get("player_id") for p in participants]
    if any(pid is None for pid in ids) or len(set(ids)) != len(participants):
        raise ValueError("Moneyball needs a fixed lineup of different players selected from the club player list, each with a starting rating.")
    response = supabase.table("players").select("id,name,rating").eq("club_id", str(club_id)).in_("id", ids).execute()
    by_id = {int(p["id"]): p for p in response.data or []}
    ratings = {}
    for p in participants:
        player = by_id.get(int(p["player_id"]))
        if player is None:
            raise ValueError("Choose every Moneyball player from this club’s player list.")
        try:
            value = float(player["rating"])/400
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or not 1 <= value <= 8:
            raise ValueError(f"{p['name']} needs a valid starting rating before Moneyball can start.")
        ratings[p["id"]] = value
    return freeze_moneyball(event, ratings)

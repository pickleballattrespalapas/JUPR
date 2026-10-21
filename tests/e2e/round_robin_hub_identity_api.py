"""Loopback-only identity browser fixture, extending the real hub route fixture.

Two distinct club profiles intentionally share a display name. All sessions,
hubs and identity decisions are created through the production HTTP routes.
One fixture-only endpoint converts a created hub to the old persisted identity
format so the browser can verify compatibility with an active court.
"""
from copy import deepcopy

from fastapi import HTTPException

from round_robin_hub_api import app, database
from jupr_app.services import admin_round_robin_hub_service as hub_service

for player_id in (101, 102):
    player = deepcopy(database.db["players"][0])
    player.update(id=player_id, name="Alex Same")
    database.db["players"].append(player)


def fixture_hub(hub_id: str):
    row = next((row for row in database.db[hub_service.HUB_TABLE] if row["id"] == hub_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown local fixture hub")
    return row


@app.get("/identity-fixture/hubs/{hub_id}")
def identity_state(hub_id: str):
    state = fixture_hub(hub_id)["state"]
    return {"identity_version": state.get("identity_version"), "player_keys": list(state.get("players") or {})}


@app.post("/identity-fixture/legacy-hubs/{hub_id}")
def convert_to_legacy(hub_id: str):
    """Simulate a persisted pre-upgrade hub without changing match/court refs."""
    state = fixture_hub(hub_id)["state"]
    legacy_players = {}
    for group in state["groups"]:
        key = group["session_key"]
        session = next(row for row in database.db["live_sessions"] if row["session_key"] == key)
        for participant in session["state"]["page_state"]["event"]["participants"]:
            identity = state["participant_identities"][key][str(participant["id"])]
            profile = participant.get("player_id")
            legacy_key = f"player:{int(profile)}" if profile is not None else f"name:{' '.join(participant['name'].lower().split())}"
            legacy_players[legacy_key] = deepcopy(state["players"][identity])
    state["players"] = legacy_players
    for key in ("identity_version", "participant_identities", "identity_decisions"):
        state.pop(key, None)
    return identity_state(hub_id)

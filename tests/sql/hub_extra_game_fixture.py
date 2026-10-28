"""Generate real Python supplemental-session output for the SQL integration test."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event
from jupr_app.domain.round_robin_hub import _ensure_players
from jupr_app.services.hub_extra_game_service import prepare_extra_game

NOW = "2026-10-28T03:00:00+00:00"
HUB_ID = "10000000-0000-4000-8000-000000000005"


class Query:
    def __init__(self, db): self.db, self.filters, self.payload = db, {}, None
    def select(self, *_): return self
    def eq(self, key, value): self.filters[key] = value; return self
    def limit(self, *_): return self
    def insert(self, payload): self.payload = deepcopy(payload); return self
    def execute(self):
        if self.payload is not None:
            self.db.rows.append({**self.payload, "version": 1})
        return SimpleNamespace(data=[deepcopy(row) for row in self.db.rows
            if all(row.get(key) == value for key, value in self.filters.items())])


class DB:
    def __init__(self): self.rows = []
    def table(self, name): assert name == "live_sessions"; return Query(self)


events = {}
for key, count in (("extra-source-low", 5), ("extra-source-high", 7)):
    events[key] = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format="doubles",
        title=key, participant_names=[f"{key} {i}" for i in range(count)], total_rounds=1, court_count=0))
lower = {"session_key": "extra-source-low", "participant_id": events["extra-source-low"]["rounds"][0]["byeParticipantIds"][0]}
higher = [{"session_key": "extra-source-high", "participant_id": pid}
    for pid in events["extra-source-high"]["rounds"][0]["byeParticipantIds"]]
next(p for p in events["extra-source-low"]["participants"] if p["id"] == lower["participant_id"])["player_id"] = 4242
hub = {"id": HUB_ID, "groups": [{"session_key": key} for key in events],
    "courts": [{"id": f"extra-c{i}", "label": f"Extra court {i}", "open": True,
        "dedicated_session_key": None, "current": None, "next": None} for i in (1, 2)], "players": {}}
_ensure_players(hub, events, NOW)
payload = {"side_a": [lower, higher[0]], "side_b": higher[1:], "standings_policy": "exhibition"}
draft, proposed, updated = prepare_extra_game(DB(), club_id="club", hub_id=HUB_ID, hub=hub, events=events,
    payload=payload, operation_key="sql-extra-operation", request_fingerprint="sql-extra-fingerprint",
    actor_email="staff@example.test", now=NOW)
print(json.dumps({"hub_id": HUB_ID, "before_hub": hub, "before_events": events,
    "draft": draft, "proposed_hub": proposed, "proposed_events": updated, "source_refs": [lower, *higher]}))

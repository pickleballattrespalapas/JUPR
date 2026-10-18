from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jupr_app.data.load import load_data
from jupr_app.domain.player_activity import add_activity_columns
from jupr_app.services.direct_match_entry_service import (
    DIRECT_MATCH_RPC,
    DirectMatchConflictError,
    submit_atomic_direct_matches,
)


MATCH_TIME = "2026-10-18T21:35:09+00:00"
LAST_GAME_AT = "2026-09-30T21:16:30.987265+00:00"
INACTIVE_AT = "2026-10-15T03:00:00.275612+00:00"
PLAYER_IDS = (991001, 991002, 991003, 991004)


class SnapshotSupabase:
    """Read fixtures and enforce the RPC's persisted player expectations."""

    def __init__(self, *, active=True, managed_league=False):
        self.tables = {
            "players": [
                {
                    "id": player_id, "club_id": "club", "name": f"Player {player_id}",
                    "rating": rating, "wins": 2, "losses": 1, "matches_played": 3,
                    "last_game_at": LAST_GAME_AT, "inactive_at": INACTIVE_AT, "active": active,
                }
                for player_id, rating in zip(PLAYER_IDS, (1299.0871, 1230.0359, 1249.9416, 1228.4286))
            ],
            "leagues_metadata": [{
                "id": 1, "club_id": "club", "league_name": "Heat league",
                "status": "active", "match_format": "doubles", "k_factor": 32,
                "is_active": True, "ended_at": None,
            }] if managed_league else [],
        }
        # Exercise the loader's merged-player filter, which copies the frame.
        self.tables["players"].append({**self.tables["players"][0], "id": 999999,
                                       "name": "Old (MERGED into Player #991001)"})
        self.calls = []

    def table(self, name):
        db = self

        class Query:
            def __init__(self):
                self.filters = []

            def select(self, *_args):
                return self

            def eq(self, field, value):
                self.filters.append(lambda row: row.get(field) == value)
                return self

            def in_(self, field, values):
                self.filters.append(lambda row: row.get(field) in values)
                return self

            def is_(self, *_args):
                return self

            def order(self, *_args, **_kwargs):
                return self

            def limit(self, *_args):
                return self

            def execute(self):
                return SimpleNamespace(data=deepcopy([
                    row for row in db.tables.get(name, [])
                    if all(predicate(row) for predicate in self.filters)
                ]))

        return Query()

    def rpc(self, name, payload):
        assert name == DIRECT_MATCH_RPC
        self.calls.append(deepcopy(payload))
        players = {row["id"]: row for row in self.tables["players"]}
        # Validate every expected value before mutating any player or match.
        for update in payload["p_player_updates"]:
            current = players[update["player_id"]]
            if any(current.get(key) != value for key, value in update["expected"].items()):
                raise RuntimeError("JUPR_DIRECT_MATCH_PLAYER_STALE: doubles state changed")
        for update in payload["p_player_updates"]:
            players[update["player_id"]].update(update["after"])
        self.tables["matches"] = deepcopy(payload["p_match_rows"])
        receipt = {
            "ok": True, "committed": True, "idempotent": False,
            "request_fingerprint": payload["p_request_fingerprint"],
            "match_ids": ["101"], "player_updates": payload["p_player_updates"],
            "result_summary": payload["p_result_summary"],
        }
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=receipt))


@pytest.fixture(autouse=True)
def isolated_side_effects(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(MATCH_TIME).astimezone(tz or timezone.utc)

    monkeypatch.setenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", "1")
    monkeypatch.setattr("jupr_app.domain.player_activity.datetime", Clock)
    monkeypatch.setattr("jupr_app.services.direct_match_entry_service._post_commit_side_effects",
                        lambda *_args, **_kwargs: ({}, {}, []))
    monkeypatch.setattr("jupr_app.services.interclub_rating_service.reconcile_interclub_for_club",
                        lambda *_args: {"status": "ok"})


def _submit(db, loaded, *, managed_league=False):
    players, _active, leagues, _matches, meta, *_rest = loaded
    # Reapplying display projection must not turn the stored snapshot into false.
    players = add_activity_columns(players.copy())
    return submit_atomic_direct_matches(
        db, club_id="club", match_format="doubles", idempotency_key="activity:regression-1",
        actor_email="admin@example.com", actor_role="admin", source="activity-regression",
        matches=[{
            "date": MATCH_TIME, "league": "Heat league" if managed_league else "Overall",
            "match_type": "League" if managed_league else "Live Match",
            "t1_p1": PLAYER_IDS[0], "t1_p2": PLAYER_IDS[1],
            "t2_p1": PLAYER_IDS[2], "t2_p2": PLAYER_IDS[3],
            "score_t1": 11, "score_t2": 7,
        }],
        name_to_id={}, df_players_all=players, df_leagues=leagues, df_meta=meta,
        queue_player_notifications=False,
    )


@pytest.mark.parametrize("stored_active", [True, None, False])
@pytest.mark.parametrize("managed_league", [False, True], ids=["standard", "league"])
def test_publish_uses_persisted_activity_and_reactivates_actual_participants(stored_active, managed_league):
    db = SnapshotSupabase(active=stored_active, managed_league=managed_league)
    original = deepcopy(db.tables["players"])
    loaded = load_data(db, "club")
    players, active_players = loaded[:2]

    assert players["id"].tolist() == list(PLAYER_IDS)
    assert players["active"].tolist() == [False] * 4
    assert active_players.empty
    assert set(players.columns) == set(original[0])
    assert set(players.to_dict("records")[0]) == set(original[0])

    result = _submit(db, loaded, managed_league=managed_league)

    assert result["match_write_committed"] is True
    assert len(db.calls) == 1
    assert len(db.tables["matches"]) == 1
    for update, before in zip(db.calls[0]["p_player_updates"], original[:4]):
        assert update["expected"] == {key: before[key] for key in update["expected"]}
        assert update["expected"]["active"] is stored_active
        assert update["expected"]["inactive_at"] == INACTIVE_AT
        assert update["after"]["active"] is True
        assert update["after"]["inactive_at"] is None
        assert update["after"]["last_game_at"] == MATCH_TIME
        assert update["after"]["matches_played"] == 4


@pytest.mark.parametrize("changed_field,changed_value", [("active", False), ("active", None), ("rating", 1300)])
def test_persisted_activity_snapshot_still_rejects_concurrent_player_changes(changed_field, changed_value):
    db = SnapshotSupabase()
    loaded = load_data(db, "club")
    db.tables["players"][0][changed_field] = changed_value
    before_submit = deepcopy(db.tables["players"])

    with pytest.raises(DirectMatchConflictError, match="Nothing from this request was written"):
        _submit(db, loaded)

    assert len(db.calls) == 1
    assert db.calls[0]["p_player_updates"][0]["expected"]["active"] is True
    assert db.tables["players"] == before_submit
    assert not db.tables.get("matches")

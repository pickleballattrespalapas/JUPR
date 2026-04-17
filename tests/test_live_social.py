from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from jupr_app.domain.live_social import (
    auto_link_exact_matches,
    find_exact_player_link_candidates,
    list_social_submissions_for_review,
    is_missing_social_tables_error,
    moderate_social_submission,
    normalized_player_name_map,
    resolve_or_create_club_person,
    save_social_live_event,
    save_social_round_robin,
    social_league_match_rows_from_event,
    social_person_rollup_rows,
    social_round_robin_match_rows_from_event,
)
from jupr_app.domain.live_social_submit import save_resolved_social_live_event
from jupr_app.ui.pages.players import fetch_player_social_event_history
import pandas as pd


class _Resp:
    def __init__(self, data):
        self.data = data


class _TableQuery:
    def __init__(self, store: dict[str, list[dict]], name: str):
        self.store = store
        self.name = name
        self._filters: list[tuple[str, object]] = []
        self._op = "select"
        self._payload = None

    def select(self, _cols: str):
        if any(col in str(_cols) for col in self.store.get("__missing_select_columns__", set())):
            raise Exception(f"PGRST204: Could not find requested column in schema cache: {_cols}")
        self._op = "select"
        return self

    def eq(self, key: str, value: object):
        self._filters.append((key, value))
        return self

    def in_(self, key: str, values: list[object]):
        self._filters.append((key, ("__in__", set(values))))
        return self

    def order(self, _key: str, desc: bool = False):
        self._order = (_key, desc)
        return self

    def limit(self, count: int):
        self._limit = int(count)
        return self

    def or_(self, _expr: str):
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict: str):
        self._op = "upsert"
        self._payload = (payload, on_conflict)
        return self

    def update(self, payload: dict):
        self._op = "update"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def _filtered(self) -> list[dict]:
        rows = list(self.store.setdefault(self.name, []))
        for key, value in self._filters:
            if isinstance(value, tuple) and value and value[0] == "__in__":
                rows = [row for row in rows if row.get(key) in value[1]]
            else:
                rows = [row for row in rows if row.get(key) == value]
        order = getattr(self, "_order", None)
        if order:
            key, desc = order
            rows = sorted(rows, key=lambda r: str(r.get(key) or ""), reverse=bool(desc))
        if getattr(self, "_limit", None) is not None:
            rows = rows[: int(self._limit)]
        return rows

    def execute(self):
        rows = self.store.setdefault(self.name, [])
        if self._op == "select":
            return _Resp([dict(r) for r in self._filtered()])
        if self._op == "delete":
            keep = []
            removed = []
            for row in rows:
                if all(row.get(k) == v for k, v in self._filters):
                    removed.append(row)
                else:
                    keep.append(row)
            self.store[self.name] = keep
            return _Resp(removed)
        if self._op == "update":
            updated = []
            for row in rows:
                if all(row.get(k) == v for k, v in self._filters):
                    row.update(dict(self._payload))
                    updated.append(dict(row))
            return _Resp(updated)
        if self._op == "insert":
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for row in payload:
                record = dict(row)
                if "id" not in record:
                    if self.name == "players":
                        existing_ids = [
                            int(existing.get("id"))
                            for existing in rows
                            if str(existing.get("id") or "").isdigit()
                        ]
                        record["id"] = (max(existing_ids) + 1) if existing_ids else 1
                    else:
                        record["id"] = str(uuid4())
                rows.append(record)
                inserted.append(dict(record))
            return _Resp(inserted)
        if self._op == "upsert":
            payload, on_conflict = self._payload
            missing_write_columns = self.store.get("__missing_write_columns__", set())
            if any(col in payload for col in missing_write_columns):
                col = next(col for col in missing_write_columns if col in payload)
                raise Exception(f"PGRST204: Could not find the '{col}' column of '{self.name}' in the schema cache")
            keys = [k.strip() for k in on_conflict.split(",")]
            match = None
            for row in rows:
                if all(row.get(k) == payload.get(k) for k in keys):
                    match = row
                    break
            if match is None:
                record = dict(payload)
                record.setdefault("id", str(uuid4()))
                rows.append(record)
                return _Resp([dict(record)])
            match.update(dict(payload))
            return _Resp([dict(match)])
        raise AssertionError(f"Unsupported op {self._op}")


class _FakeSupabase:
    def __init__(self):
        self.store: dict[str, list[dict]] = {}
        self.calls: list[str] = []

    def table(self, name: str):
        self.calls.append(name)
        return _TableQuery(self.store, name)


@dataclass
class _Ctx:
    supabase: _FakeSupabase
    club_id: str
    name_to_id: dict


def _sample_event() -> dict:
    return {
        "type": "round_robin",
        "name": "Friday Social",
        "sourceEventUid": "rr-fixed-1",
        "eventDate": "2026-03-29",
        "participants": [
            {"id": "p-1", "name": "Alice", "seed": 1},
            {"id": "p-2", "name": "Bob", "seed": 2},
            {"id": "p-3", "name": "Cami", "seed": 3},
            {"id": "p-4", "name": "Drew", "seed": 4},
        ],
        "rounds": [
            {
                "number": 1,
                "matches": [
                    {"id": "m1", "teamA": ["p-1", "p-2"], "teamB": ["p-3", "p-4"], "scoreA": 11, "scoreB": 7},
                    {"id": "m2", "teamA": ["p-1", "p-3"], "teamB": ["p-2", "p-4"], "scoreA": None, "scoreB": None},
                ],
            }
        ],
    }


def _sample_league_event() -> dict:
    return {
        "type": "league",
        "name": "Ladder Night",
        "sourceEventUid": "lg-fixed-1",
        "eventDate": "2026-03-29",
        "currentRoundNumber": 2,
        "totalRounds": 3,
        "participants": [
            {"id": "p-1", "name": "Alice", "seed": 1},
            {"id": "p-2", "name": "Bob", "seed": 2},
            {"id": "p-3", "name": "Cami", "seed": 3},
            {"id": "p-4", "name": "Drew", "seed": 4},
        ],
        "rounds": [
            {
                "number": 1,
                "courts": [
                    {
                        "courtNumber": 1,
                        "miniRounds": [
                            {
                                "number": 1,
                                "matches": [
                                    {
                                        "id": "lg-r1-m1",
                                        "teamA": ["p-1", "p-2"],
                                        "teamB": ["p-3", "p-4"],
                                        "scoreA": 11,
                                        "scoreB": 9,
                                        "courtNumber": 1,
                                        "miniRoundNumber": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "number": 2,
                "courts": [
                    {
                        "courtNumber": 1,
                        "miniRounds": [
                            {
                                "number": 1,
                                "matches": [
                                    {
                                        "id": "lg-r2-m1",
                                        "teamA": ["p-1", "p-3"],
                                        "teamB": ["p-2", "p-4"],
                                        "scoreA": 11,
                                        "scoreB": 5,
                                        "courtNumber": 1,
                                        "miniRoundNumber": 1,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_match_rows_only_emit_scored_matches():
    rows = social_round_robin_match_rows_from_event(_sample_event())

    assert len(rows) == 1
    assert rows[0]["match_key"] == "m1"


def test_resolve_or_create_links_existing_player_row():
    supabase = _FakeSupabase()
    supabase.store["club_people"] = [
        {
            "id": "cp-1",
            "club_id": "club-1",
            "display_name": "Alice",
            "normalized_name": "alice",
            "linked_player_id": 10,
            "first_seen_on": "2026-03-01",
            "last_seen_on": "2026-03-10",
        }
    ]
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={"Alice": 10})

    row, created_new, matched_player = resolve_or_create_club_person(
        ctx,
        display_name="Alice",
        event_date="2026-03-29",
    )

    assert row["id"] == "cp-1"
    assert created_new is False
    assert matched_player is True
    assert supabase.store["club_people"][0]["last_seen_on"] == "2026-03-29"


def test_resolve_or_create_creates_social_person_when_unmatched():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})

    row, created_new, matched_player = resolve_or_create_club_person(
        ctx,
        display_name="New Person",
        event_date="2026-03-29",
    )

    assert row["linked_player_id"] is None
    assert row["normalized_name"] == "new person"
    assert created_new is True
    assert matched_player is False


def test_resave_replaces_children_instead_of_duplicating():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={"Alice": 10})
    event = _sample_event()

    first = save_social_round_robin(ctx, event)
    assert first["match_count"] == 1

    event["rounds"][0]["matches"][0]["scoreA"] = 11
    event["rounds"][0]["matches"][0]["scoreB"] = 2
    second = save_social_round_robin(ctx, event)

    assert first["event_id"] == second["event_id"]
    participants = [
        row for row in supabase.store["live_event_participants"] if row["event_id"] == first["event_id"]
    ]
    matches = [
        row for row in supabase.store["live_event_matches"] if row["event_id"] == first["event_id"]
    ]
    assert len(participants) == 4
    assert len(matches) == 1
    assert matches[0]["score_t2"] == 2


def test_social_league_rows_emit_only_scored_matches():
    event = _sample_league_event()
    event["rounds"][1]["courts"][0]["miniRounds"][0]["matches"][0]["scoreA"] = None
    event["rounds"][1]["courts"][0]["miniRounds"][0]["matches"][0]["scoreB"] = None
    rows = social_league_match_rows_from_event(event)
    assert [row["match_key"] for row in rows] == ["lg-r1-m1"]


def test_public_social_submission_saves_pending():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    event = _sample_event()
    result = save_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="public",
        host_name="Court Host",
    )
    assert result["status"] == "pending"
    assert result["submission_mode"] == "public"
    assert result["submitted_by_name"] == "Court Host"
    assert result["saved_rounds"] == ["rr"]
    assert supabase.store["live_events"][0]["status"] == "pending"


def test_admin_social_submission_saves_saved_status():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    event = _sample_event()
    result = save_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    assert result["status"] == "saved"
    assert result["submission_mode"] == "admin"
    assert result["submitted_by_name"] == "admin"
    assert supabase.store["live_events"][0]["status"] == "saved"


def test_league_resave_replaces_children_and_persists_scored_rounds():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    event = _sample_league_event()
    first = save_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host 1",
    )
    assert first["match_count"] == 2
    assert first["saved_rounds"] == [1, 2]
    event["rounds"][1]["courts"][0]["miniRounds"][0]["matches"][0]["scoreB"] = 8
    second = save_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host 1",
    )
    assert second["event_id"] == first["event_id"]
    assert second["status"] == "pending"
    matches = [
        row for row in supabase.store["live_event_matches"] if row["event_id"] == first["event_id"]
    ]
    assert len(matches) == 2
    updated = [row for row in matches if row["match_key"] == "lg-r2-m1"][0]
    assert updated["score_t2"] == 8


def test_social_save_does_not_write_public_matches_table():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    assert "matches" not in supabase.calls


def test_resolved_social_save_handles_submitted_by_schema_drift():
    supabase = _FakeSupabase()
    supabase.store["__missing_write_columns__"] = {"submitted_by"}
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    result = save_resolved_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    assert result["status"] == "saved"
    assert len(supabase.store["live_events"]) == 1
    assert supabase.store["live_events"][0]["submitted_by_name"] == "admin"


def test_resolved_save_preserves_explicit_existing_player_linkage():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    event = _sample_event()
    event["participants"][0]["name"] = "Alias Name"
    event["participants"][0]["player_id"] = 77
    event["participants"][0]["match_status"] = "matched_existing"
    save_resolved_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    saved = supabase.store["live_event_participants"]
    linked_row = [row for row in saved if row["participant_key"] == "p-1"][0]
    assert linked_row["linked_player_id"] == 77


def test_resolved_save_creates_new_rated_player_and_links_participant():
    supabase = _FakeSupabase()
    supabase.store["players"] = [
        {"id": 101, "club_id": "club-1", "name": "Alice", "rating": 1600.0, "starting_rating": 1600.0},
        {"id": 102, "club_id": "club-1", "name": "Bob", "rating": 1200.0, "starting_rating": 1200.0},
    ]
    ctx = _Ctx(
        supabase=supabase,
        club_id="club-1",
        name_to_id={},
    )
    ctx.admin_logged_in = True
    ctx.df_players_all = pd.DataFrame(
        [
            {"id": 101, "name": "Alice", "rating": 1600.0},
            {"id": 102, "name": "Bob", "rating": 1200.0},
        ]
    )
    event = _sample_event()
    event["participants"] = [
        {"id": "p-1", "name": "Alice", "player_id": 101, "seed": 1, "match_status": "matched_existing"},
        {"id": "p-2", "name": "Bob", "player_id": 102, "seed": 2, "match_status": "matched_existing"},
        {"id": "p-3", "name": "New Rated", "player_id": None, "seed": 3, "match_status": "create_rated"},
        {"id": "p-4", "name": "Drew", "player_id": None, "seed": 4, "match_status": "new_social"},
    ]
    result = save_resolved_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    assert result["created_rated_players_count"] == 1
    created_players = [row for row in supabase.store["players"] if row["name"] == "New Rated"]
    assert len(created_players) == 1
    created = created_players[0]
    assert created["rating"] == 1400.0
    unchanged_alice = [row for row in supabase.store["players"] if row["id"] == 101][0]
    unchanged_bob = [row for row in supabase.store["players"] if row["id"] == 102][0]
    assert unchanged_alice["rating"] == 1600.0
    assert unchanged_bob["rating"] == 1200.0
    participants = supabase.store["live_event_participants"]
    created_participant = [row for row in participants if row["participant_key"] == "p-3"][0]
    assert created_participant["linked_player_id"] == created["id"]


def test_resolved_save_falls_back_to_default_provisional_seed_without_other_rated_players():
    supabase = _FakeSupabase()
    supabase.store["players"] = []
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    ctx.admin_logged_in = True
    ctx.df_players_all = pd.DataFrame()
    event = _sample_event()
    for participant in event["participants"]:
        participant["player_id"] = None
        participant["match_status"] = "create_rated"
    save_resolved_social_live_event(
        ctx,
        event,
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    created = [row for row in supabase.store["players"] if row["club_id"] == "club-1"]
    assert all(float(row["rating"]) == 1400.0 for row in created)


def test_resolved_save_detects_strong_duplicate_and_blocks_creation():
    supabase = _FakeSupabase()
    supabase.store["players"] = [
        {"id": 55, "club_id": "club-1", "name": "Jon Snow", "rating": 1500.0, "starting_rating": 1500.0}
    ]
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    ctx.admin_logged_in = True
    ctx.df_players_all = pd.DataFrame([{"id": 55, "name": "Jon Snow", "rating": 1500.0}])
    event = _sample_event()
    event["participants"][0]["name"] = "John Snow"
    event["participants"][0]["player_id"] = None
    event["participants"][0]["match_status"] = "create_rated"
    try:
        save_resolved_social_live_event(
            ctx,
            event,
            target_club_id="club-1",
            submission_mode="admin",
            host_name="admin",
        )
    except ValueError as exc:
        assert "Duplicate warning" in str(exc)
    else:
        raise AssertionError("Expected duplicate warning ValueError")


def test_social_history_falls_back_to_legacy_submitted_by_column():
    fetch_player_social_event_history.clear()
    supabase = _FakeSupabase()
    supabase.store["__missing_select_columns__"] = {"submitted_by_name"}
    supabase.store["live_events"] = [
        {
            "id": "evt-1",
            "club_id": "club-1",
            "name": "Social Night",
            "event_type": "round_robin",
            "event_date": "2026-03-29",
            "submitted_by": "Legacy Host",
            "status": "saved",
            "result_mode": "social_unrated",
            "summary_json": {},
        }
    ]
    supabase.store["live_event_participants"] = [
        {"id": "lep-1", "event_id": "evt-1", "club_person_id": "cp-1", "linked_player_id": 12}
    ]
    supabase.store["live_event_matches"] = []
    df = fetch_player_social_event_history(supabase, "club-1", 12, limit=20)
    assert len(df) == 1
    assert df.iloc[0]["Submitted By"] == "Legacy Host"


def test_resolved_save_status_admin_vs_public():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    ctx.admin_logged_in = True
    admin_result = save_resolved_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="admin",
        host_name="admin",
    )
    assert admin_result["status"] == "saved"
    public_result = save_resolved_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    assert public_result["status"] == "pending"


def test_approve_moves_pending_to_saved():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    save_result = save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    moderated = moderate_social_submission(
        ctx,
        event_id=save_result["event_id"],
        action="approve",
    )
    assert moderated["status"] == "saved"
    assert moderated["rejection_reason"] is None
    assert moderated["moderated_at"] is not None


def test_reject_moves_pending_to_rejected():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    save_result = save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    moderated = moderate_social_submission(
        ctx,
        event_id=save_result["event_id"],
        action="reject",
        rejection_reason="spam test",
    )
    assert moderated["status"] == "rejected"
    assert moderated["rejection_reason"] == "spam test"
    assert moderated["moderated_at"] is not None


def test_public_resubmission_resets_rejected_event_to_pending_and_clears_reason():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    save_result = save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    moderate_social_submission(
        ctx,
        event_id=save_result["event_id"],
        action="reject",
        rejection_reason="bad data",
    )
    resaved = save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    assert resaved["status"] == "pending"
    row = supabase.store["live_events"][0]
    assert row["status"] == "pending"
    assert row["rejection_reason"] is None


def test_list_review_queue_scopes_to_club_and_status():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})
    other_ctx = _Ctx(supabase=supabase, club_id="club-2", name_to_id={})
    save_social_live_event(
        ctx,
        _sample_event(),
        target_club_id="club-1",
        submission_mode="public",
        host_name="Host",
    )
    save_social_live_event(
        other_ctx,
        _sample_event(),
        target_club_id="club-2",
        submission_mode="public",
        host_name="Host",
    )
    queue = list_social_submissions_for_review(ctx, status="pending", limit=10)
    assert len(queue) == 1
    assert queue[0]["club_id"] == "club-1"


def test_exact_match_auto_link_only_links_unambiguous_matches():
    players = pd.DataFrame(
        [
            {"id": 10, "name": "Alice"},
            {"id": 11, "name": "Bob"},
        ]
    )
    club_people = [
        {"id": "cp-1", "display_name": "Alice", "normalized_name": "alice", "linked_player_id": None},
        {"id": "cp-2", "display_name": "No Match", "normalized_name": "no match", "linked_player_id": None},
    ]
    player_map = normalized_player_name_map(players)
    matches = find_exact_player_link_candidates(club_people, player_map)
    assert matches == {"cp-1": 10}


def test_exact_match_auto_link_skips_ambiguous_same_names():
    players = pd.DataFrame(
        [
            {"id": 10, "name": "Sam"},
            {"id": 11, "name": "Sam"},
        ]
    )
    club_people = [{"id": "cp-1", "display_name": "Sam", "normalized_name": "sam", "linked_player_id": None}]
    player_map = normalized_player_name_map(players)
    matches = find_exact_player_link_candidates(club_people, player_map)
    assert matches == {}


def test_auto_link_exact_matches_updates_database_rows():
    supabase = _FakeSupabase()
    supabase.store["club_people"] = [
        {"id": "cp-1", "club_id": "club-1", "display_name": "Alice", "normalized_name": "alice", "linked_player_id": None}
    ]
    players = pd.DataFrame([{"id": 10, "name": "Alice"}])
    result = auto_link_exact_matches(
        supabase,
        club_id="club-1",
        club_people_rows=supabase.store["club_people"],
        df_players_all=players,
    )
    assert result["linked_count"] == 1
    assert supabase.store["club_people"][0]["linked_player_id"] == 10


def test_social_person_rollup_rows_counts_events_and_matches():
    supabase = _FakeSupabase()
    supabase.store["club_people"] = [
        {"id": "cp-1", "club_id": "club-1", "display_name": "Alice", "normalized_name": "alice", "linked_player_id": None},
    ]
    supabase.store["live_event_participants"] = [
        {"id": "p-1", "event_id": "evt-1", "club_person_id": "cp-1"},
        {"id": "p-2", "event_id": "evt-2", "club_person_id": "cp-1"},
    ]
    supabase.store["live_event_matches"] = [
        {
            "id": "m-1",
            "t1_p1_participant_id": "p-1",
            "t1_p2_participant_id": "p-1",
            "t2_p1_participant_id": "p-1",
            "t2_p2_participant_id": "p-1",
            "score_t1": 11,
            "score_t2": 8,
        }
    ]
    rows = social_person_rollup_rows(supabase, "club-1")
    assert rows[0]["social_event_count"] == 2
    assert rows[0]["social_match_count"] == 1


class _MissingTableError(Exception):
    def __init__(self):
        super().__init__("PGRST205: Could not find the table 'public.club_people' in the schema cache")
        self.code = "PGRST205"


def test_detects_missing_social_tables_error_codes():
    assert is_missing_social_tables_error(_MissingTableError()) is True


def test_ignores_unrelated_errors_for_social_table_detection():
    err = RuntimeError("network timeout")
    assert is_missing_social_tables_error(err) is False

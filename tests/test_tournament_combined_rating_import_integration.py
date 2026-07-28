from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_tournament_registration_import_service import (
    import_admin_tournament_registrations_to_draw,
)


class _Query:
    def __init__(self, tables, table_name):
        self.tables = tables
        self.table_name = table_name
        self.filters = []
        self.limit_value = None
        self.insert_payload = None
        self.delete_mode = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def delete(self):
        self.delete_mode = True
        return self

    def execute(self):
        table = self.tables.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = (
                self.insert_payload
                if isinstance(self.insert_payload, list)
                else [self.insert_payload]
            )
            table.extend(dict(row) for row in rows)
            return SimpleNamespace(data=rows)
        matched = [
            row
            for row in table
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        if self.limit_value is not None:
            matched = matched[: self.limit_value]
        if self.delete_mode:
            self.tables[self.table_name] = [
                row for row in table if row not in matched
            ]
        return SimpleNamespace(data=matched)


class CombinedImportSupabase:
    def __init__(self, tables):
        self.tables = tables
        self.rpc_calls: list[dict] = []

    def table(self, name):
        return _Query(self.tables, name)

    def rpc(self, name, params):
        self.rpc_calls.append({"name": name, "params": dict(params)})

        def execute():
            assert name == "admin_write_combined_rating_draw_teams_cas"
            saved = [dict(row) for row in params["p_teams"]]
            if params["p_replace"]:
                self.tables["tournament_teams"] = saved
            else:
                self.tables.setdefault("tournament_teams", []).extend(saved)
            return SimpleNamespace(
                data={
                    "ok": True,
                    "teams": saved,
                    "operation_key": params["p_operation_key"],
                }
            )

        return SimpleNamespace(execute=execute)


def _combined_tables() -> dict:
    return {
        "tournaments": [
            {
                "id": "tournament-1",
                "club_id": "club-1",
                "name": "Combined Cup",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw-1",
                "tournament_id": "tournament-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "updated_at": "2026-07-27T12:00:00Z",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event-1",
                "tournament_id": "tournament-1",
                "eligibility_mode": "COMBINED_RATING_CAP",
                "combined_rating_cap": 8,
            }
        ],
        "tournament_registrations": [
            {
                "id": "registration-1",
                "tournament_id": "tournament-1",
                "email": "one@example.com",
                "display_name": "One",
                "status": "confirmed",
                "player_id": 1,
            },
            {
                "id": "registration-2",
                "tournament_id": "tournament-1",
                "email": "two@example.com",
                "display_name": "Two",
                "status": "confirmed",
                "player_id": 2,
            },
        ],
        "tournament_registration_selections": [
            {
                "id": "selection-1",
                "tournament_id": "tournament-1",
                "registration_id": "registration-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                # Browser-entered partner text is not authoritative import
                # evidence. The finalized review below owns the relationship.
                "partner_email": "attacker@example.com",
                "player_id": 999,
                "partner_player_id": 998,
            }
        ],
        "tournament_rating_eligibility_reviews": [
            {
                "id": "review-1",
                "tournament_id": "tournament-1",
                "event_option_id": "event-1",
                "selection_id": "selection-1",
                "registration_id": "registration-1",
                "partner_registration_id": "registration-2",
                "player_id_snapshot": 1,
                "partner_player_id_snapshot": 2,
                "review_phase": "REGISTRATION_CLOSE",
                "state": "ELIGIBLE",
                "override_state": None,
                "finalized_at": "2026-07-27T11:00:00Z",
            }
        ],
        "tournament_games": [],
        "tournament_teams": [],
        "admin_activity_log": [],
    }


def test_combined_rating_import_uses_guarded_snapshot_rpc_and_stable_replay(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = _combined_tables()
    supabase = CombinedImportSupabase(tables)
    kwargs = {
        "club_id": "club-1",
        "tournament_id": "tournament-1",
        "draw_id": "draw-1",
        "import_mode": "REPLACE",
        "actor_email": "admin@example.com",
        "actor_role": "club_owner",
        "confirmation_text": "IMPORT REGISTRATIONS",
        "expected_draw_updated_at": "2026-07-27T12:00:00Z",
        "atomic": True,
    }

    first = import_admin_tournament_registrations_to_draw(supabase, **kwargs)
    # A response-loss retry must reach the database operation replay before
    # current draw-activity guards. A genuinely new operation is still blocked
    # by the RPC's in-transaction game/podium check.
    tables["tournament_games"].append(
        {
            "id": "scheduled-after-commit",
            "tournament_id": "tournament-1",
            "draw_id": "draw-1",
        }
    )
    second = import_admin_tournament_registrations_to_draw(supabase, **kwargs)

    assert first["teams"][0]["source_selection_id"] == "selection-1"
    assert first["teams"][0]["source"] == "REGISTRATION_COMBINED_RATING"
    assert second["teams"][0]["source_selection_id"] == "selection-1"
    assert [call["name"] for call in supabase.rpc_calls] == [
        "admin_write_combined_rating_draw_teams_cas",
        "admin_write_combined_rating_draw_teams_cas",
    ]
    first_params = supabase.rpc_calls[0]["params"]
    second_params = supabase.rpc_calls[1]["params"]
    assert first_params["p_teams"][0]["source_selection_id"] == "selection-1"
    assert first_params["p_teams"][0]["player1_id"] == 1
    assert first_params["p_teams"][0]["player2_id"] == 2
    assert first_params["p_operation_key"] == second_params["p_operation_key"]
    assert (
        first_params["p_request_fingerprint"]
        == second_params["p_request_fingerprint"]
    )
    assert tables["admin_activity_log"][-1]["action_type"] == (
        "import_tournament_registration_teams_admin"
    )


def test_combined_rating_import_dry_run_exposes_source_without_writing(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = _combined_tables()
    supabase = CombinedImportSupabase(tables)

    result = import_admin_tournament_registrations_to_draw(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        draw_id="draw-1",
        import_mode="REPLACE",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="IMPORT REGISTRATIONS",
        expected_draw_updated_at="2026-07-27T12:00:00Z",
        dry_run=True,
        atomic=True,
    )

    assert result["dry_run"] is True
    assert result["write_count"] == 0
    assert result["teams"][0]["source_selection_id"] == "selection-1"
    assert supabase.rpc_calls == []
    assert tables["tournament_teams"] == []

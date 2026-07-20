from jupr_app.domain import tournament_registration_repo as repo
from jupr_app.domain.tournament_public_references import build_public_tournament_reference


def test_public_roster_includes_all_registered_statuses_and_needs_partner_only_list(monkeypatch):
    compiled_state = {
        "event_options": [
            {
                "id": "event-1",
                "event_family_label": "Men's Doubles",
                "division_name": "3.5",
            },
            {
                "id": "event-2",
                "event_family_label": "Mixed Doubles",
                "division_name": "4.0",
            },
        ],
        "event_rosters": [
            {
                "event_option_id": "event-1",
                "event_day_id": "day-3",
                "event_day_label": "Day 3",
                "event_label": "Men's Doubles 3.5",
                "entries": [
                    {
                        "status": "CONFIRMED",
                        "members": [{"display_name": "Confirmed Player"}],
                        "source_selection_ids": ["sel-confirmed"],
                    },
                    {
                        "status": "WAITLIST",
                        "members": [{"display_name": "Waitlist Player"}],
                        "source_selection_ids": ["sel-waitlist"],
                    },
                    {
                        "status": "NEEDS_PARTNER",
                        "show_on_partner_board": False,
                        "members": [
                            {
                                "registration_id": "reg-joe",
                                "selection_id": "sel-joe",
                                "player_id": 42,
                                "display_name": "Joe Baumann",
                                "skill": "3.5",
                                "age": 42,
                                "age_bracket": "40+",
                            }
                        ],
                        "notes": "Can play either side",
                        "source_selection_ids": ["sel-joe"],
                    },
                    {
                        "status": "REVIEW",
                        "members": [{"display_name": "Review Player"}],
                        "source_selection_ids": ["sel-review"],
                    },
                    {
                        "status": "PARTNER_MISSING",
                        "members": [{"display_name": "Partner Missing Player"}],
                        "source_selection_ids": ["sel-missing"],
                    },
                    {
                        "status": None,
                        "members": [{"display_name": "Unknown State Player"}],
                        "source_selection_ids": ["sel-unknown"],
                    },
                    {
                        "status": "CONFIRMED",
                        "members": [],
                    },
                ],
            },
            {
                "event_option_id": "event-2",
                "event_day_id": "day-4-5",
                "event_day_label": "Day 4/5",
                "event_label": "Mixed Doubles 4.0",
                "entries": [
                    {
                        "status": "NEEDS_PARTNER",
                        "members": [
                            {
                                "display_name": "Mixed Needs Partner",
                                "registration_id": "reg-mixed",
                                "selection_id": "sel-mixed",
                                "player_id": 88,
                            }
                        ],
                        "source_selection_ids": ["sel-mixed"],
                    },
                ],
            },
        ],
        "summary": {"total_registrations": 6, "waitlist_entries": 1},
    }

    monkeypatch.setattr(
        repo,
        "build_registration_state",
        lambda supabase, tournament, settings, days, event_options: compiled_state,
    )

    state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])

    roster_rows = state["registrations_by_event"]
    roster_names = [
        member["display_name"]
        for row in roster_rows
        for member in row["members"]
    ]
    assert roster_names == [
        "Confirmed Player",
        "Waitlist Player",
        "Joe Baumann",
        "Review Player",
        "Partner Missing Player",
        "Unknown State Player",
        "Mixed Needs Partner",
    ]
    assert [row["status"] for row in roster_rows] == [
        "Registered",
        "Waitlist",
        "Needs Partner",
        "Review",
        "Review",
        "Review",
        "Needs Partner",
    ]
    assert "Pending Review" not in {row["status"] for row in roster_rows}

    needs_partner_row = next(
        row
        for row in roster_rows
        if row["members"][0]["display_name"] == "Joe Baumann"
    )
    assert needs_partner_row["event_day_label"] == "Day 3"
    assert needs_partner_row["event_family"] == "Men's Doubles"
    assert needs_partner_row["division"] == "3.5"
    assert needs_partner_row["status"] == "Needs Partner"

    assert state["players_needing_partners"] == [
        {
            "player_name": "Joe Baumann",
            "board_entry_key": build_public_tournament_reference(
                tournament_id="t-1",
                namespace="partner-board-selection",
                source_id="sel-joe",
            ),
            "event_day_label": "Day 3",
            "event_family": "Men's Doubles",
            "division": "3.5",
            "event_label": "Men's Doubles 3.5",
            "skill": "3.5",
            "age_bracket": "40+",
            "note": "Can play either side",
        },
        {
            "player_name": "Mixed Needs Partner",
            "board_entry_key": build_public_tournament_reference(
                tournament_id="t-1",
                namespace="partner-board-selection",
                source_id="sel-mixed",
            ),
            "event_day_label": "Day 4/5",
            "event_family": "Mixed Doubles",
            "division": "4.0",
            "event_label": "Mixed Doubles 4.0",
            "skill": None,
            "age_bracket": None,
            "note": "",
        },
    ]
    assert state["summary"]["waitlist"] == 1
    assert [row["members"][0]["display_name"] for row in state["confirmed_teams"]] == ["Confirmed Player"]
    assert state["pending_partner_requests"] == []
    assert [row["members"][0]["display_name"] for row in state["unresolved_partner_entries"]] == ["Review Player", "Partner Missing Player", "Unknown State Player"]


def test_public_roster_null_state_and_email_fallback_fail_closed(monkeypatch):
    compiled_state = {
        "event_options": [{"id": "event-1", "event_family_label": "Doubles", "division_name": "Open"}],
        "event_rosters": [
            {
                "event_option_id": "event-1",
                "event_day_label": "Saturday",
                "event_label": "Open Doubles",
                "entries": [
                    {
                        "status": None,
                        "members": [
                            {
                                "display_name": "private@example.com",
                                "email": "private@example.com",
                                "phone": "555-010-1234",
                                "dupr_id": "PRIVATE-DUPR",
                                "player_id": 99,
                                "age": 34,
                            }
                        ],
                        "source_registration_ids": ["reg-private"],
                        "source_selection_ids": ["sel-private"],
                        "source_player_ids": [99],
                    }
                ],
            }
        ],
        "summary": {},
    }
    monkeypatch.setattr(repo, "build_registration_state", lambda *_args, **_kwargs: compiled_state)

    state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])
    row = state["registrations_by_event"][0]

    assert row["status"] == "Review"
    assert row["members"] == [{"display_name": "Player", "skill": None, "age_bracket": "30-39"}]
    assert row["public_entry_key"].startswith("tr_")
    assert "source_selection_ids" not in row
    assert "registration_id" not in row["members"][0]
    assert state["confirmed_teams"] == []
    assert state["unresolved_partner_entries"] == [row]


def test_partner_board_projection_requires_display_and_contact_consent(monkeypatch):
    compiled_state = {
        "settings": {"partner_board_enabled": True},
        "event_options": [
            {
                "id": "event-1",
                "event_family_label": "Doubles",
                "division_name": "Open",
                "enabled": True,
                "partner_board_enabled": True,
                "status": "open",
            }
        ],
        "registrations": [
            {"id": "reg-visible", "wants_partner_board_contact": True, "status": "CONFIRMED"},
            {"id": "reg-private", "wants_partner_board_contact": False, "status": "CONFIRMED"},
        ],
        "partner_board": [
            {"selection_id": "sel-visible", "registration_id": "reg-visible", "event_option_id": "event-1"},
            {"selection_id": "sel-private", "registration_id": "reg-private", "event_option_id": "event-1"},
        ],
        "event_rosters": [
            {
                "event_option_id": "event-1",
                "event_day_id": "day-1",
                "event_day_label": "Saturday",
                "event_label": "Open Doubles",
                "entries": [
                    {
                        "status": "NEEDS_PARTNER",
                        "members": [{"registration_id": "reg-visible", "selection_id": "sel-visible", "display_name": "Visible Player"}],
                    },
                    {
                        "status": "NEEDS_PARTNER",
                        "members": [{"registration_id": "reg-private", "selection_id": "sel-private", "display_name": "Private Player"}],
                    },
                ],
            }
        ],
        "summary": {"total_registrations": 2},
    }
    monkeypatch.setattr(repo, "build_registration_state", lambda *_args, **_kwargs: compiled_state)

    state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])

    assert [row["player_name"] for row in state["players_needing_partners"]] == ["Visible Player", "Private Player"]
    assert [row["player_name"] for row in state["partner_board_entries"]] == ["Visible Player"]
    assert state["summary"]["partner_board_entries"] == 1
    assert "email" not in str(state["partner_board_entries"]).lower()
    assert "phone" not in str(state["partner_board_entries"]).lower()

    compiled_state["settings"]["partner_board_enabled"] = False
    disabled_state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])
    assert disabled_state["partner_board_entries"] == []

    compiled_state["settings"]["partner_board_enabled"] = True
    compiled_state["event_options"][0]["partner_board_enabled"] = False
    disabled_event_state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])
    assert disabled_event_state["partner_board_entries"] == []

from types import SimpleNamespace


class _SchemaQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.limit_n = None
        self.selected_columns = "*"
        self.order_column = None

    def select(self, columns, *args, **kwargs):
        self.selected_columns = columns
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def limit(self, value):
        self.limit_n = int(value)
        return self

    def order(self, column, *args, **kwargs):
        self.order_column = column
        return self

    def execute(self):
        if self.table_name not in self.storage:
            raise RuntimeError(f"relation {self.table_name} does not exist")
        rows = self.storage[self.table_name]
        wanted = [column.strip() for column in str(self.selected_columns).split(",") if column.strip() and column.strip() != "*"]
        for column in wanted:
            for row in rows[:1]:
                if column not in row:
                    raise RuntimeError(f"column {self.table_name}.{column} does not exist")
        data = [dict(row) for row in rows if all(str(row.get(column)) == str(value) for column, value in self.filters)]
        if self.order_column:
            data.sort(key=lambda row: str(row.get(self.order_column) or ""))
        if self.limit_n is not None:
            data = data[: self.limit_n]
        return SimpleNamespace(data=data, count=len(data))


class _SchemaSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return _SchemaQuery(self.storage, name)


def _core_storage(*, include_partner_schema=False, include_core_selection_table=True, include_player_id=False):
    registration = {
        "id": "reg-mary",
        "tournament_id": "tour-1",
        "display_name": "Mary Bauman",
        "email": "mary@example.com",
        "submitted_at": "2026-01-01T00:00:00",
        "doubles_skill": 3.5,
    }
    if include_player_id:
        registration["player_id"] = 101
    storage = {
        "tournament_registration_settings": [
            {"id": "settings-1", "tournament_id": "tour-1", "builder_draft_json": {}, "builder_draft_updated_at": "now"}
        ],
        "tournament_registration_days": [
            {"id": "day-1", "tournament_id": "tour-1", "enabled": True, "label": "Day 1", "sort_order": 1}
        ],
        "tournament_event_options": [
            {
                "id": "event-wd-35",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "event_family_label": "Women's Doubles",
                "division_name": "3.5",
                "event_format_default": "Round Robin",
                "scoring_default": "Games",
                "event_format_override": None,
                "scoring_override": None,
                "skill_mode": "division",
                "age_mode": "none",
                "age_rules": {},
                "waitlist_enabled": True,
                "partner_board_enabled": True,
                "status": "open",
                "enabled": True,
                "event_type": "GENDER_DOUBLES",
                "label": "Women's Doubles 3.5",
            }
        ],
        "tournament_registrations": [registration],
    }
    if include_core_selection_table:
        storage["tournament_registration_selections"] = [
            {
                "id": "sel-mary",
                "tournament_id": "tour-1",
                "registration_id": "reg-mary",
                "registration_day_id": "day-1",
                "event_option_id": "event-wd-35",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Elizabeth Whelan",
                "partner_email": "elizabeth@example.com",
                "created_at": "2026-01-01T00:00:00",
            }
        ]
    if include_partner_schema:
        storage["tournament_registration_partner_requests"] = []
        storage["tournament_registration_team_links"] = []
        storage["tournament_registration_team_members"] = []
    return storage


def test_core_registration_available_when_partner_link_tables_missing():
    supabase = _SchemaSupabase(_core_storage(include_partner_schema=False))

    available, detail = repo.registration_feature_available(supabase)
    partner_available, partner_detail = repo.partner_link_schema_available(supabase)
    state = repo.build_registration_state(
        supabase,
        {"id": "tour-1"},
        {},
        supabase.storage["tournament_registration_days"],
        supabase.storage["tournament_event_options"],
    )

    assert available is True
    assert detail is None
    assert partner_available is False
    assert "tournament_registration_partner_requests" in partner_detail
    assert state["partner_link_schema_available"] is False
    assert [entry["status"] for row in state["event_rosters"] for entry in row["entries"]] == ["LEGACY_PARTNER_UNRESOLVED"]


def test_missing_partner_link_tables_do_not_confirm_free_text_partner():
    supabase = _SchemaSupabase(_core_storage(include_partner_schema=False))

    state = repo.build_public_tournament_roster_state(
        supabase,
        {"id": "tour-1"},
        {},
        supabase.storage["tournament_registration_days"],
        supabase.storage["tournament_event_options"],
    )

    assert state["confirmed_teams"] == []
    assert [row["members"][0]["display_name"] for row in state["unresolved_partner_entries"]] == ["Mary Bauman"]


def test_partner_link_schema_available_when_tables_and_player_id_exist():
    supabase = _SchemaSupabase(_core_storage(include_partner_schema=True, include_player_id=True))

    available, detail = repo.partner_link_schema_available(supabase)

    assert available is True
    assert detail is None


def test_registration_feature_unavailable_when_core_table_missing():
    supabase = _SchemaSupabase(_core_storage(include_core_selection_table=False))

    available, detail = repo.registration_feature_available(supabase)

    assert available is False
    assert "tournament_registration_selections" in detail

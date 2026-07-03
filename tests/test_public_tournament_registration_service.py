from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_tournament_registration_service import (
    build_public_tournament_registration_confirmation,
    build_public_tournament_registration_page,
    submit_public_tournament_registration,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.limit_count: int | None = None
        self.order_key: str | None = None
        self.order_desc = False
        self.insert_payload = None
        self.upsert_payload = None
        self.update_payload = None
        self.delete_mode = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append(("eq", key, value))
        return self

    def neq(self, key, value):
        self.filters.append(("neq", key, value))
        return self

    def in_(self, key, values):
        self.filters.append(("in", key, set(values or [])))
        return self

    def limit(self, value):
        self.limit_count = int(value)
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self.upsert_payload = payload
        self.on_conflict = on_conflict
        return self

    def update(self, payload):
        self.update_payload = dict(payload or {})
        return self

    def delete(self):
        self.delete_mode = True
        return self

    def _apply_filters(self, rows):
        result = list(rows)
        for op, key, expected in self.filters:
            if op == "eq":
                result = [row for row in result if str(row.get(key)) == str(expected)]
            elif op == "neq":
                result = [row for row in result if str(row.get(key)) != str(expected)]
            elif op == "in":
                result = [row for row in result if row.get(key) in expected]
        return result

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            for row in rows:
                table.append(dict(row))
            return SimpleNamespace(data=rows, count=len(rows))
        if self.upsert_payload is not None:
            rows = self.upsert_payload if isinstance(self.upsert_payload, list) else [self.upsert_payload]
            keys = [part.strip() for part in str(getattr(self, "on_conflict", "id") or "id").split(",") if part.strip()]
            for row in rows:
                clean = dict(row)
                idx = next((i for i, existing in enumerate(table) if all(str(existing.get(key)) == str(clean.get(key)) for key in keys)), None)
                if idx is None:
                    table.append(clean)
                else:
                    table[idx].update(clean)
            return SimpleNamespace(data=rows, count=len(rows))
        matched = self._apply_filters(table)
        if self.delete_mode:
            remaining = [row for row in table if row not in matched]
            self.storage[self.table_name] = remaining
            return SimpleNamespace(data=matched, count=len(matched))
        if self.update_payload is not None:
            for row in matched:
                row.update(self.update_payload)
            return SimpleNamespace(data=matched, count=len(matched))
        if self.order_key:
            matched = sorted(matched, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_count is not None:
            matched = matched[: self.limit_count]
        return SimpleNamespace(data=matched, count=len(matched))


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "tournaments": [
            {
                "id": "t1",
                "club_id": "club-1",
                "name": "Tres Palapas Open",
                "status": "DRAFT",
                "start_date": "2026-09-01",
                "created_at": "2026-01-01T00:00:00Z",
                "admin_notes": "private",
            }
        ],
        "tournament_registration_settings": [
            {
                "id": "rs1",
                "tournament_id": "t1",
                "registration_slug": "tres-open",
                "registration_status": "open",
                "waitlist_enabled": True,
                "partner_board_enabled": True,
                "rules_markdown": "Be kind.",
                "refund_policy_markdown": "No refunds after draw publication.",
                "builder_draft_json": {"private": True},
                "builder_draft_updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        "tournament_registration_days": [
            {"id": "day1", "tournament_id": "t1", "sort_order": 1, "label": "Saturday", "event_date": "2026-09-01", "enabled": True}
        ],
        "tournament_event_options": [
            {
                "id": "event1",
                "tournament_id": "t1",
                "registration_day_id": "day1",
                "sort_order": 1,
                "label": "Open Doubles",
                "event_family_label": "Doubles",
                "division_name": "Open",
                "event_type": "DOUBLES",
                "gender_restriction": "ANY",
                "skill_label": "Open",
                "age_label": "All ages",
                "event_format_default": "ROUND_ROBIN_PLUS_PLAYOFF",
                "scoring_default": "GAME_TO_15",
                "skill_mode": "OPEN",
                "age_mode": "ALL_AGES",
                "waitlist_enabled": True,
                "partner_board_enabled": True,
                "status": "open",
                "enabled": True,
                "partner_required": False,
                "capacity_teams": 16,
                "price_usd": 50,
                "internal_seed_notes": "private",
            }
        ],
        "tournament_registrations": [],
        "tournament_registration_selections": [],
        "tournament_registration_partner_requests": [],
        "tournament_registration_team_links": [],
        "tournament_registration_team_members": [],
        "tournament_event_draws": [],
        "tournament_teams": [],
        "tournament_games": [],
    }


def test_public_tournament_registration_page_is_public_safe() -> None:
    storage = fake_storage()
    payload = build_public_tournament_registration_page(FakeSupabase(storage), club_id="club-1", registration_slug="tres-open")

    assert payload["available"] is True
    assert payload["registration_open"] is True
    assert payload["tournament"]["name"] == "Tres Palapas Open"
    assert payload["events"][0]["selectable"] is True
    assert payload["events"][0]["price_usd"] == 50
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]
    assert "builder_draft_json" not in payload["settings"]


def test_public_tournament_registration_submit_and_confirmation() -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert result["ok"] is True
    assert result["selection_count"] == 1
    assert len(storage["tournament_registrations"]) == 1
    assert len(storage["tournament_registration_selections"]) == 1

    confirmation = build_public_tournament_registration_confirmation(
        supabase,
        club_id="club-1",
        registration_id=result["registration_id"],
        registration_slug="tres-open",
    )
    assert confirmation is not None
    assert confirmation["registration"]["display_name"] == "Alex Rivera"
    assert confirmation["selections"][0]["event_label"] == "Open"
    assert "phone" not in confirmation["registration"]


def test_public_tournament_registration_blocks_honeypot() -> None:
    try:
        submit_public_tournament_registration(
            FakeSupabase(fake_storage()),
            club_id="club-1",
            payload={"website": "bot", "terms_accepted": True, "email": "bot@example.com", "first_name": "Bot", "selections": [{"event_option_id": "event1"}]},
        )
    except ValueError as exc:
        assert "Unable to submit" in str(exc)
    else:
        raise AssertionError("Expected honeypot submission to fail")

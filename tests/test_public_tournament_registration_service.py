from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.domain.tournament_registration_repo import PUBLIC_REGISTRATION_EDIT_RPC
from jupr_app.services.public_tournament_registration_service import (
    DuplicateTournamentRegistrationError,
    build_registration_confirmation_delivery,
    build_public_tournament_registration_confirmation,
    build_public_tournament_registration_page,
    resolve_public_tournament_registration_profile,
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
        self.rpc_calls: list[tuple[str, dict]] = []

    def table(self, name):
        return FakeQuery(self.storage, name)

    def rpc(self, name, params):
        self.rpc_calls.append((str(name), deepcopy(dict(params or {}))))
        return FakeRegistrationEditRpc(self.storage, str(name), dict(params or {}))


class FakeRegistrationEditRpc:
    def __init__(self, storage, name, params):
        self.storage = storage
        self.name = name
        self.params = params

    def execute(self):
        if self.name != PUBLIC_REGISTRATION_EDIT_RPC:
            raise AssertionError(f"Unexpected fake RPC: {self.name}")
        tournament_id = str(self.params.get("p_tournament_id") or "")
        registration_id = str(self.params.get("p_registration_id") or "")
        registration = next(
            (
                row
                for row in self.storage.get("tournament_registrations", [])
                if str(row.get("tournament_id")) == tournament_id and str(row.get("id")) == registration_id
            ),
            None,
        )
        if not registration:
            return SimpleNamespace(data={"ok": False, "code": "REGISTRATION_NOT_FOUND"})
        if str(registration.get("updated_at") or "") != str(self.params.get("p_expected_updated_at") or ""):
            return SimpleNamespace(data={"ok": False, "code": "REGISTRATION_EDIT_CONFLICT"})

        current_selections = [
            row
            for row in self.storage.get("tournament_registration_selections", [])
            if str(row.get("tournament_id")) == tournament_id and str(row.get("registration_id")) == registration_id
        ]
        expected_versions = {
            str(row.get("id") or ""): str(row.get("updated_at") or "")
            for row in self.params.get("p_expected_selection_versions") or []
        }
        current_versions = {
            str(row.get("id") or ""): str(row.get("updated_at") or "")
            for row in current_selections
        }
        if expected_versions != current_versions:
            return SimpleNamespace(data={"ok": False, "code": "REGISTRATION_EDIT_CONFLICT"})

        imported_pairs = {
            (str(row.get("registration_day_id") or ""), str(row.get("event_option_id") or ""))
            for row in self.storage.get("tournament_teams", [])
            if str(row.get("tournament_id")) == tournament_id and str(row.get("source") or "").upper() == "REGISTRATION"
        }
        if any(
            (str(row.get("registration_day_id") or ""), str(row.get("event_option_id") or "")) in imported_pairs
            for row in current_selections
        ):
            return SimpleNamespace(data={"ok": False, "code": "REGISTRATION_IMPORTED_TO_DRAW"})

        if self.storage.get("_fail_public_registration_edit_rpc"):
            raise RuntimeError("simulated atomic RPC failure")

        next_version = "2099-01-01T00:00:00.000001Z"
        registration.update(deepcopy(self.params.get("p_registration_patch") or {}))
        registration["updated_at"] = next_version
        retained = [
            row
            for row in self.storage.get("tournament_registration_selections", [])
            if not (
                str(row.get("tournament_id")) == tournament_id
                and str(row.get("registration_id")) == registration_id
            )
        ]
        replacements = []
        current_by_id = {str(row.get("id") or ""): row for row in current_selections}
        for index, desired in enumerate(self.params.get("p_selections") or []):
            selection_id = str(desired.get("id") or "")
            row = deepcopy(current_by_id.get(selection_id) or {})
            row.update(deepcopy(desired))
            row.update(
                {
                    "id": selection_id,
                    "tournament_id": tournament_id,
                    "registration_id": registration_id,
                    "sort_order": index,
                    "updated_at": next_version,
                }
            )
            replacements.append(row)
        self.storage["tournament_registration_selections"] = [*retained, *replacements]
        return SimpleNamespace(
            data={
                "ok": True,
                "registration_id": registration_id,
                "updated_at": next_version,
                "selection_count": len(replacements),
            }
        )


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
                "sponsor_markdown": "Presented by Rally House.",
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
    assert payload["settings"]["sponsor_markdown"] == "Presented by Rally House."


def test_public_profile_resolution_returns_safe_suggestion_without_linking_policy() -> None:
    storage = fake_storage()
    storage["players"] = [
        {
            "id": 10,
            "club_id": "club-1",
            "name": "Avery Ace",
            "email": "avery@example.com",
            "phone": "private",
            "rating": 1600,
            "dupr_id": "DUPR-10",
            "active": True,
            "inactive_at": None,
        }
    ]

    result = resolve_public_tournament_registration_profile(
        FakeSupabase(storage),
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Avery",
            "last_name": "Ace",
            "email": "avery@example.com",
            "age": 34,
            "gender": "Women",
        },
    )

    assert result["status"] == "ready"
    assert result["profile_match_kind"] == "email_exact"
    assert result["profile_candidates"] == [
        {
            "id": "10",
            "display_name": "Avery Ace",
            "dupr_id": "DUPR-10",
            "doubles_skill": 4.0,
            "singles_skill": 4.0,
        }
    ]
    assert result["profile_policy"] == {
        "linkage": "staff_review_required",
        "public_submission_links_player": False,
    }
    assert "phone" not in result["profile_candidates"][0]
    assert "email" not in result["profile_candidates"][0]


def test_public_profile_resolution_hands_existing_email_to_recovery_without_id() -> None:
    storage = fake_storage()
    storage["tournament_registrations"] = [
        {"id": "private-registration-id", "tournament_id": "t1", "email": "avery@example.com"}
    ]

    result = resolve_public_tournament_registration_profile(
        FakeSupabase(storage),
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Avery",
            "last_name": "Ace",
            "email": "avery@example.com",
            "age": 34,
            "gender": "Women",
        },
    )

    assert result["status"] == "existing_registration"
    assert result["can_start_new"] is False
    assert result["masked_email"] == "a***y@example.com"
    assert "private-registration-id" not in str(result)


def test_public_profile_resolution_reports_closed_state_before_profile_discovery() -> None:
    storage = fake_storage()
    storage["tournament_registration_settings"][0]["registration_status"] = "closed"

    result = resolve_public_tournament_registration_profile(
        FakeSupabase(storage),
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Avery",
            "last_name": "Ace",
            "email": "avery@example.com",
            "age": 34,
            "gender": "Women",
        },
    )

    assert result["status"] == "closed"
    assert result["registration_open"] is False
    assert result["profile_candidates"] == []


def test_public_tournament_registration_submit_and_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "unit-test-confirmation-secret")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://staging.example.test")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
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
    assert result["confirmation_token"]
    assert result["email_delivery"]["status"] == "dry_run"

    confirmation = build_public_tournament_registration_confirmation(
        supabase,
        club_id="club-1",
        confirmation_token=result["confirmation_token"],
    )
    assert confirmation is not None
    assert confirmation["registration"]["display_name"] == "Alex Rivera"
    assert confirmation["selections"][0]["event_label"] == "Open"
    assert confirmation["selections"][0]["price_usd"] == 50
    assert "phone" not in confirmation["registration"]
    assert "email" not in confirmation["registration"]
    assert "id" not in confirmation["registration"]
    assert "selection_id" not in confirmation["selections"][0]


def test_completed_registration_replay_builds_token_without_resending_email(
    monkeypatch,
) -> None:
    storage = fake_storage()
    storage["tournament_registrations"].append(
        {
            "id": "reg_replay",
            "tournament_id": "t1",
            "submitted_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-01T00:00:00Z",
            "status": "confirmed",
            "payment_status": "unpaid",
            "display_name": "Replay Player",
            "email": "replay@example.com",
        }
    )
    monkeypatch.setenv(
        "JUPR_REGISTRATION_CONFIRMATION_SECRET",
        "unit-test-confirmation-secret",
    )
    monkeypatch.setattr(
        "jupr_app.services.public_tournament_registration_service."
        "send_tournament_registration_confirmation_email",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed replay must not resend email")
        ),
    )

    result = build_registration_confirmation_delivery(
        FakeSupabase(storage),
        club_id="club-1",
        club_slug="tres-palapas",
        tournament_id="t1",
        registration_id="reg_replay",
        send_email=False,
    )

    assert result["confirmation_available"] is True
    assert result["confirmation_token"]
    assert result["email_delivery"]["status"] == "already_completed"


def test_registration_stays_saved_when_confirmation_email_fails(monkeypatch) -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "unit-test-confirmation-secret")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://staging.example.test")
    monkeypatch.setattr(
        "jupr_app.services.public_tournament_registration_service.send_tournament_registration_confirmation_email",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated delivery failure")),
    )

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Saved",
            "last_name": "Player",
            "email": "saved@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert result["ok"] is True
    assert result["confirmation_token"]
    assert result["email_delivery"]["status"] == "failed"
    assert any(row["email"] == "saved@example.com" for row in storage["tournament_registrations"])
    assert len(storage["tournament_registration_selections"]) == 1


def test_registration_stays_saved_when_confirmation_reload_fails(monkeypatch) -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "unit-test-confirmation-secret")
    monkeypatch.setattr(
        "jupr_app.services.public_tournament_registration_service.get_registration_confirmation_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated reload failure")),
    )

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Saved",
            "last_name": "Before Reload",
            "email": "saved-before-reload@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert result["confirmation_available"] is False
    assert result["email_delivery"]["status"] == "failed"
    assert any(row["email"] == "saved-before-reload@example.com" for row in storage["tournament_registrations"])
    assert len(storage["tournament_registration_selections"]) == 1


def test_confirmation_link_requires_explicit_next_web_origin(monkeypatch) -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "unit-test-confirmation-secret")
    monkeypatch.setenv("JUPR_PUBLIC_BASE_URL", "https://streamlit.example.test")
    for name in ("JUPR_WEB_BASE_URL", "STAGING_WEB_BASE_URL", "NEXT_PUBLIC_JUPR_WEB_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "jupr_app.services.public_tournament_registration_service.send_tournament_registration_confirmation_email",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("email should not be attempted")),
    )

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Saved",
            "last_name": "Without Next Origin",
            "email": "saved-no-origin@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert result["confirmation_token"]
    assert result["email_delivery"]["status"] == "failed"
    assert "link is not configured" in result["email_delivery"]["message"]
    assert any(row["email"] == "saved-no-origin@example.com" for row in storage["tournament_registrations"])


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


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"age": None, "gender": "Women"}, "Age is required"),
        ({"age": 34, "gender": ""}, "Gender is required"),
    ],
)
def test_fastapi_public_contract_requires_age_and_gender(patch, message) -> None:
    payload = {
        "registration_slug": "tres-open",
        "first_name": "Alex",
        "last_name": "Rivera",
        "email": "alex@example.com",
        "age": 34,
        "gender": "Women",
        "terms_accepted": True,
        "_require_demographics": True,
        "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        **patch,
    }

    with pytest.raises(ValueError, match=message):
        submit_public_tournament_registration(FakeSupabase(fake_storage()), club_id="club-1", payload=payload)


def test_duplicate_email_refuses_second_write_and_requires_recovery() -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    payload = {
        "registration_slug": "tres-open",
        "first_name": "Alex",
        "last_name": "Rivera",
        "email": "alex@example.com",
        "age": 34,
        "gender": "Men",
        "terms_accepted": True,
        "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
    }
    submit_public_tournament_registration(supabase, club_id="club-1", payload=payload)

    with pytest.raises(DuplicateTournamentRegistrationError, match="secure edit-link"):
        submit_public_tournament_registration(supabase, club_id="club-1", payload=payload)

    assert len(storage["tournament_registrations"]) == 1
    assert len(storage["tournament_registration_selections"]) == 1


def test_partner_required_and_partner_board_consent_integrity() -> None:
    storage = fake_storage()
    storage["tournament_event_options"][0].update(
        {"event_type": "GENDER_DOUBLES", "partner_required": True, "division_name": "Doubles"}
    )
    base = {
        "registration_slug": "tres-open",
        "first_name": "Alex",
        "last_name": "Rivera",
        "email": "alex@example.com",
        "age": 34,
        "gender": "Women",
        "terms_accepted": True,
    }

    with pytest.raises(ValueError, match="choose whether you have or need a partner"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={**base, "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}]},
        )

    with pytest.raises(ValueError, match="partner age is required"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                **base,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "HAS_PARTNER",
                        "partner_name": "Casey Court",
                        "partner_email": "casey@example.com",
                        "partner_gender": "Men",
                    }
                ],
            },
        )

    with pytest.raises(ValueError, match="Partner-board contact consent"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                **base,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "NEEDS_PARTNER",
                        "show_on_partner_board": True,
                    }
                ],
            },
        )


def test_registered_partner_without_demographics_uses_submitted_review_fields() -> None:
    storage = fake_storage()
    storage["tournament_event_options"][0].update(
        {"event_type": "GENDER_DOUBLES", "partner_required": True, "division_name": "Doubles"}
    )
    storage["tournament_registrations"] = [
        {
            "id": "partner-registration",
            "tournament_id": "t1",
            "email": "casey@example.com",
            "display_name": "Casey Court",
        }
    ]

    result = submit_public_tournament_registration(
        FakeSupabase(storage),
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "age": 34,
            "gender": "Women",
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "HAS_PARTNER",
                    "partner_name": "Casey Court",
                    "partner_email": "casey@example.com",
                    "partner_age": 35,
                    "partner_gender": "Men",
                    "partner_skill": 3.5,
                }
            ],
        },
    )

    assert result["ok"] is True
    saved = storage["tournament_registration_selections"][-1]
    assert saved["partner_age"] == 35
    assert saved["partner_skill"] == 3.5


def test_singles_rejects_partner_fields() -> None:
    storage = fake_storage()
    storage["tournament_event_options"][0].update(
        {"event_type": "SINGLES", "partner_required": False, "division_name": "Singles"}
    )

    with pytest.raises(ValueError, match="does not accept partner information"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                "registration_slug": "tres-open",
                "first_name": "Alex",
                "email": "alex@example.com",
                "age": 34,
                "gender": "Women",
                "terms_accepted": True,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "HAS_PARTNER",
                        "partner_name": "Casey",
                        "partner_email": "casey@example.com",
                        "partner_age": 35,
                        "partner_gender": "Men",
                    }
                ],
            },
        )


def test_public_registration_hides_players_and_initial_player_link_is_not_trusted() -> None:
    storage = fake_storage()
    storage["players"] = [
        {"id": 10, "club_id": "club-1", "name": "Verified Avery", "email": "avery@example.com", "dupr_id": "canonical-dupr", "rating": 1600, "active": True, "inactive_at": None},
        {"id": 11, "club_id": "other-club", "name": "Other Club", "rating": 800, "active": True, "inactive_at": None},
        {"id": 12, "club_id": "club-1", "name": "Inactive Player", "rating": 800, "active": False, "inactive_at": "2026-01-01"},
    ]
    supabase = FakeSupabase(storage)

    page = build_public_tournament_registration_page(supabase, club_id="club-1", registration_slug="tres-open")
    assert page["players"] == []

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Avery",
            "email": "avery@example.com",
            "player_id": 10,
            "doubles_skill": 1.0,
            "singles_skill": 1.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    registration = next(row for row in storage["tournament_registrations"] if row["id"] == result["registration_id"])
    assert registration["player_id"] is None
    assert registration["dupr_id"] is None
    assert registration["doubles_skill"] == 1.0
    assert registration["singles_skill"] == 1.0

    untrusted = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Mallory",
            "email": "mallory@example.com",
            "player_id": 11,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    untrusted_registration = next(
        row for row in storage["tournament_registrations"] if row["id"] == untrusted["registration_id"]
    )
    assert untrusted_registration["player_id"] is None


def test_public_registration_rejects_multiple_divisions_in_same_day_family() -> None:
    storage = fake_storage()
    storage["tournament_event_options"].append(
        {
            **storage["tournament_event_options"][0],
            "id": "event2",
            "label": "Advanced Doubles",
            "division_name": "Advanced",
            "sort_order": 2,
        }
    )

    with pytest.raises(ValueError, match="only one division"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                "registration_slug": "tres-open",
                "first_name": "Alex",
                "email": "alex@example.com",
                "terms_accepted": True,
                "selections": [
                    {"event_option_id": "event1", "partner_mode": "NONE"},
                    {"event_option_id": "event2", "partner_mode": "NONE"},
                ],
            },
        )


def test_public_registration_enforces_division_gender_and_rating() -> None:
    storage = fake_storage()
    event = storage["tournament_event_options"][0]
    event.update(
        {
            "event_type": "SINGLES",
            "partner_required": False,
            "gender_restriction": "WOMEN",
            "skill_label": "3.5",
            "division_name": "Women's 3.5",
        }
    )

    with pytest.raises(ValueError, match="women's registrations"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                "registration_slug": "tres-open",
                "first_name": "Alex",
                "email": "alex@example.com",
                "gender": "Men",
                "singles_skill": 3.2,
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )

    with pytest.raises(ValueError, match="above the 3.5 division cap"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                "registration_slug": "tres-open",
                "first_name": "Casey",
                "email": "casey@example.com",
                "gender": "Women",
                "singles_skill": 4.0,
                "terms_accepted": True,
                "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
            },
        )


def test_public_registration_enforces_partner_identity_gender_and_rating() -> None:
    storage = fake_storage()
    event = storage["tournament_event_options"][0]
    event.update(
        {
            "event_type": "MIXED_DOUBLES",
            "partner_required": True,
            "gender_restriction": "MIXED",
            "skill_label": "3.5",
            "division_name": "Mixed 3.5",
        }
    )
    base = {
        "registration_slug": "tres-open",
        "first_name": "Alex",
        "email": "alex@example.com",
        "gender": "Women",
        "doubles_skill": 3.2,
        "terms_accepted": True,
    }

    with pytest.raises(ValueError, match="choose whether you have or need a partner"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={**base, "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}]},
        )

    with pytest.raises(ValueError, match="their own partner"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                **base,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "HAS_PARTNER",
                        "partner_name": "Alex",
                        "partner_email": "alex@example.com",
                        "partner_age": 35,
                        "partner_gender": "Men",
                    }
                ],
            },
        )

    with pytest.raises(ValueError, match="one men's and one women's"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                **base,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "HAS_PARTNER",
                        "partner_name": "Pat",
                        "partner_email": "pat@example.com",
                        "partner_age": 35,
                        "partner_gender": "Women",
                        "partner_skill": 3.2,
                    }
                ],
            },
        )

    with pytest.raises(ValueError, match="above the 3.5 division cap"):
        submit_public_tournament_registration(
            FakeSupabase(storage),
            club_id="club-1",
            payload={
                **base,
                "selections": [
                    {
                        "event_option_id": "event1",
                        "partner_mode": "HAS_PARTNER",
                        "partner_name": "Pat",
                        "partner_email": "pat@example.com",
                        "partner_age": 35,
                        "partner_gender": "Men",
                        "partner_skill": 4.0,
                    }
                ],
            },
        )

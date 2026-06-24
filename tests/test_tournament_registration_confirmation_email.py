from decimal import Decimal

from jupr_app.domain.notifications import tournament_registration_confirmation_email as emailer


def _vm():
    return emailer.build_registration_confirmation_view_model(
        tournament={"name": "Tres Palapas Open"},
        registration={"id": "reg_1", "display_name": "Ada Lovelace", "email": "ada@example.com"},
        selections=[
            {"registration_day_id": "day_1", "event_option_id": "event_1", "partner_mode": "HAS_PARTNER", "partner_name": "Grace"},
            {"registration_day_id": "day_2", "event_option_id": "event_2", "partner_mode": "NEEDS_PARTNER"},
        ],
        days=[{"id": "day_1", "label": "Saturday"}, {"id": "day_2", "label": "Sunday"}],
        event_options=[
            {"id": "event_1", "registration_day_id": "day_1", "event_family_label": "Women's Doubles", "division_name": "3.5", "price_usd": 40},
            {"id": "event_2", "registration_day_id": "day_2", "event_family_label": "Mixed Doubles", "division_name": "4.0", "price_usd": "40"},
        ],
        confirmation_url="https://example.test/?page=tournament_registration_confirmation",
        sender_from_name="JUPR Notifications",
        sender_from_email="noreply@example.com",
    )


def test_build_view_model_two_events_total_80():
    vm = _vm()
    assert vm["total_price_usd"] == Decimal("80")
    assert len(vm["selected_events"]) == 2


def test_missing_blank_price_treated_as_zero():
    vm = emailer.build_registration_confirmation_view_model(
        tournament={"name": "No Price Open"},
        registration={"display_name": "Player", "email": "p@example.com"},
        selections=[{"event_option_id": "e1"}, {"event_option_id": "e2"}],
        event_options=[{"id": "e1", "price_usd": ""}, {"id": "e2"}],
    )
    assert vm["total_price_usd"] == Decimal("0")
    assert emailer.format_money(vm["total_price_usd"]) == "$0"


def test_html_email_contains_required_details():
    html = emailer.build_tournament_registration_confirmation_html(_vm())
    assert "Tres Palapas Open" in html
    assert "Ada Lovelace" in html
    assert "Women" in html and "Doubles" in html
    assert "Mixed Doubles" in html
    assert "Total due: $80" in html
    assert emailer.PAYMENT_NOTE in html


def test_text_email_contains_event_list_total_and_payment_note():
    text = emailer.build_tournament_registration_confirmation_text(_vm())
    assert "Events:" in text
    assert "Women's Doubles" in text
    assert "Mixed Doubles" in text
    assert "Total due: $80" in text
    assert emailer.PAYMENT_NOTE in text


def test_sender_from_address_in_view_model_when_provided():
    vm = _vm()
    assert vm["sender_from_name"] == "JUPR Notifications"
    assert vm["sender_from_email"] == "noreply@example.com"


def test_dry_run_send_path_does_not_call_smtp(monkeypatch):
    monkeypatch.setattr(emailer, "get_email_mode", lambda: emailer.EMAIL_MODE_DRY_RUN)
    monkeypatch.setattr(emailer, "send_email_with_inline_chart", lambda **kwargs: (_ for _ in ()).throw(AssertionError("smtp called")))
    result = emailer.send_tournament_registration_confirmation_email(view_model=_vm())
    assert result["status"] == "dry_run"
    assert result["provider_message_id"] == "dry_run"


def test_staging_redirect_sends_to_redirect_and_prefixes_subject(monkeypatch):
    calls = []
    monkeypatch.setattr(emailer, "get_email_mode", lambda: emailer.EMAIL_MODE_STAGING_REDIRECT)
    monkeypatch.setenv("JUPR_STAGING_EMAIL_REDIRECT_TO", "staff@example.com")
    monkeypatch.setattr(emailer, "send_email_with_inline_chart", lambda **kwargs: calls.append(kwargs) or "smtp")
    result = emailer.send_tournament_registration_confirmation_email(view_model=_vm())
    assert result["to_email"] == "staff@example.com"
    assert calls[0]["to_email"] == "staff@example.com"
    assert calls[0]["subject"].startswith("[STAGING→ada@example.com]")


def test_live_send_path_uses_registrant_email(monkeypatch):
    calls = []
    monkeypatch.setattr(emailer, "get_email_mode", lambda: emailer.EMAIL_MODE_LIVE)
    monkeypatch.setattr(emailer, "send_email_with_inline_chart", lambda **kwargs: calls.append(kwargs) or "smtp")
    result = emailer.send_tournament_registration_confirmation_email(view_model=_vm())
    assert result["status"] == "sent"
    assert calls[0]["to_email"] == "ada@example.com"
    assert calls[0]["chart_png_bytes"] is None


class _CaptureQuery:
    def __init__(self, table_name, storage, calls):
        self.table_name = table_name
        self.storage = storage
        self.calls = calls
        self.filters = []

    def upsert(self, row, on_conflict=None):
        self.calls.append(("upsert", self.table_name, row, on_conflict))
        self.storage[self.table_name] = [row]
        return self

    def delete(self):
        self.calls.append(("delete", self.table_name))
        self.storage[self.table_name] = []
        return self

    def insert(self, rows):
        self.calls.append(("insert", self.table_name, rows))
        self.storage.setdefault(self.table_name, []).extend(rows)
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        self.calls.append(("eq", self.table_name, column, value))
        return self

    def execute(self):
        return type("Resp", (), {"data": self.storage.get(self.table_name, [])})()


class _CaptureSupabase:
    def __init__(self):
        self.storage = {}
        self.calls = []

    def table(self, name):
        return _CaptureQuery(name, self.storage, self.calls)


def test_non_edit_duplicate_registration_does_not_silently_overwrite(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: {"id": "reg_existing", "email": "ada@example.com"})

    import pytest
    with pytest.raises(ValueError, match="secure edit link"):
        repo.save_registration(
            supabase,
            tournament_id="tournament_1",
            payload={"display_name": "Ada Lovelace", "email": "ADA@EXAMPLE.COM", "selections": []},
        )


def test_edit_registration_updates_expected_id_and_replaces_selections(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: {"id": "reg_existing", "email": "ada@example.com"})
    monkeypatch.setattr(repo, "get_registration_by_id", lambda *_args: {"id": "reg_existing", "email": "ada@example.com", "tournament_id": "tournament_1"})
    monkeypatch.setattr(repo, "list_registration_days", lambda *_args: [{"id": "day_1", "enabled": True}])
    monkeypatch.setattr(repo, "list_event_options", lambda *_args: [{"id": "event_1", "registration_day_id": "day_1", "status": "open", "enabled": True, "division_name": "3.5"}])
    monkeypatch.setattr(repo, "validate_selection_against_skill", lambda **_kwargs: (True, None))

    result = repo.save_registration(
        supabase,
        tournament_id="tournament_1",
        expected_registration_id="reg_existing",
        payload={"display_name": "Ada Lovelace", "email": "ADA@EXAMPLE.COM", "selections": [{"event_option_id": "event_1", "registration_day_id": "day_1"}]},
    )

    assert result["registration_id"] == "reg_existing"
    assert supabase.storage["tournament_registrations"][0]["email"] == "ada@example.com"
    assert ("delete", "tournament_registration_selections") in supabase.calls


def test_edit_registration_mismatched_expected_id_raises(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo
    import pytest

    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: {"id": "reg_existing", "email": "ada@example.com"})
    monkeypatch.setattr(repo, "get_registration_by_id", lambda *_args: {"id": "other", "email": "other@example.com", "tournament_id": "tournament_1"})

    with pytest.raises(ValueError, match="email cannot be changed"):
        repo.save_registration(_CaptureSupabase(), tournament_id="tournament_1", expected_registration_id="other", payload={"display_name": "Ada", "email": "ada@example.com", "selections": []})


def test_normal_new_registration_with_unused_email_creates(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: None)
    monkeypatch.setattr(repo, "list_registration_days", lambda *_args: [])
    monkeypatch.setattr(repo, "list_event_options", lambda *_args: [])

    result = repo.save_registration(supabase, tournament_id="tournament_1", payload={"display_name": "New Player", "email": "new@example.com", "selections": []})
    assert result["registration_id"].startswith("reg_")
    assert supabase.storage["tournament_registrations"][0]["email"] == "new@example.com"


def test_new_registration_without_status_defaults_confirmed(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: None)
    monkeypatch.setattr(repo, "list_registration_days", lambda *_args: [])
    monkeypatch.setattr(repo, "list_event_options", lambda *_args: [])

    repo.save_registration(
        supabase,
        tournament_id="tournament_1",
        payload={"display_name": "Confirmed Player", "email": "confirmed@example.com", "selections": []},
    )

    assert supabase.storage["tournament_registrations"][0]["status"] == "confirmed"


def test_admin_supplied_registration_status_is_respected(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: None)
    monkeypatch.setattr(repo, "list_registration_days", lambda *_args: [])
    monkeypatch.setattr(repo, "list_event_options", lambda *_args: [])

    repo.save_registration(
        supabase,
        tournament_id="tournament_1",
        payload={"display_name": "Waitlisted Player", "email": "waitlist@example.com", "status": "waitlist", "selections": []},
    )
    assert supabase.storage["tournament_registrations"][0]["status"] == "waitlist"

    repo.save_registration(
        supabase,
        tournament_id="tournament_1",
        payload={"display_name": "Cancelled Player", "email": "cancelled@example.com", "status": "cancelled", "selections": []},
    )
    assert supabase.storage["tournament_registrations"][0]["status"] == "cancelled"


def test_edit_registration_without_status_preserves_existing_status(monkeypatch):
    from jupr_app.domain import tournament_registration_repo as repo

    supabase = _CaptureSupabase()
    existing = {"id": "reg_existing", "email": "ada@example.com", "tournament_id": "tournament_1", "status": "waitlist"}
    monkeypatch.setattr(repo, "_get_existing_registration_by_email", lambda *_args: existing)
    monkeypatch.setattr(repo, "get_registration_by_id", lambda *_args: existing)
    monkeypatch.setattr(repo, "list_registration_days", lambda *_args: [])
    monkeypatch.setattr(repo, "list_event_options", lambda *_args: [])

    repo.save_registration(
        supabase,
        tournament_id="tournament_1",
        expected_registration_id="reg_existing",
        payload={"display_name": "Ada Lovelace", "email": "ADA@EXAMPLE.COM", "selections": []},
    )

    assert supabase.storage["tournament_registrations"][0]["status"] == "waitlist"

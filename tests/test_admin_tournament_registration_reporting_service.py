from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.admin_tournament_registration_reporting_service import (
    build_admin_tournament_broadcast_preview,
    build_admin_tournament_registration_export,
)
from tests.test_admin_match_log_service import FakeQuery, FakeSupabase
from tests.test_api_contract_admin_tournament import tournament_tables


def test_registration_export_filters_by_event_and_payment(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    result = build_admin_tournament_registration_export(
        FakeSupabase(tournament_tables()),
        club_id="club",
        tournament_id="tour_1",
        payment_status="paid",
        event_option_id="event_1",
    )

    assert result["row_count"] == 1
    assert result["rows"][0]["division"] == "Gender Doubles / 3.5"
    assert result["rows"][0]["submitted_at"] == "2026-03-03T00:00:00Z"
    assert "alex@example.com" in result["csv"]


def test_registration_export_neutralizes_spreadsheet_formulas(monkeypatch):
    tables = tournament_tables()
    tables["tournament_registrations"][0]["display_name"] = "  =2+2"
    tables["tournament_registrations"][0]["phone"] = "+15550100"
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    result = build_admin_tournament_registration_export(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour_1",
    )

    assert "'=2+2" in result["csv"]
    assert "'+15550100" in result["csv"]
    assert "\n  =2+2," not in result["csv"]


def test_broadcast_preview_deduplicates_multi_selection_email(monkeypatch):
    tables = tournament_tables()
    tables["tournament_registration_selections"].append(
        {
            "id": "selection_2",
            "tournament_id": "tour_1",
            "registration_id": "registration_1",
            "registration_day_id": "day_1",
            "event_option_id": "event_2",
            "partner_mode": "NONE",
        }
    )
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    result = build_admin_tournament_broadcast_preview(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour_1",
        subject="Update",
        message="Hello",
    )

    assert result["recipient_count"] == 1
    assert result["recipient_csv"].count("alex@example.com") == 1
    assert result["dry_run"] is True
    assert result["send_available"] is False


def test_registration_reporting_rejects_cross_club_tournament(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    with pytest.raises(ValueError, match="tournament not found"):
        build_admin_tournament_registration_export(
            FakeSupabase(tournament_tables()),
            club_id="another-club",
            tournament_id="tour_1",
        )


class _BrokenRegistrationQuery(FakeQuery):
    def execute(self):
        raise RuntimeError("database unavailable")


class _BrokenRegistrationSupabase(FakeSupabase):
    def table(self, name):
        if name == "tournament_registrations":
            return _BrokenRegistrationQuery(self.tables, name)
        return super().table(name)


def test_registration_export_fails_closed_on_database_error(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    with pytest.raises(RuntimeError, match="Could not load tournament registrations"):
        build_admin_tournament_registration_export(
            _BrokenRegistrationSupabase(tournament_tables()),
            club_id="club",
            tournament_id="tour_1",
        )


class _TruncatedRegistrationQuery(FakeQuery):
    def execute(self):
        response = super().execute()
        return SimpleNamespace(data=response.data, count=len(response.data) + 1)


class _TruncatedRegistrationSupabase(FakeSupabase):
    def table(self, name):
        if name == "tournament_registrations":
            return _TruncatedRegistrationQuery(self.tables, name)
        return super().table(name)


def test_registration_export_fails_closed_on_truncated_result(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")

    with pytest.raises(RuntimeError, match="result was truncated"):
        build_admin_tournament_registration_export(
            _TruncatedRegistrationSupabase(tournament_tables()),
            club_id="club",
            tournament_id="tour_1",
        )

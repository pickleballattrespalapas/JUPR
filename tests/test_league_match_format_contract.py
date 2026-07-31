from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.match_processing import build_active_league_metadata_expectations
from jupr_app.services.admin_match_uploader_service import build_admin_match_uploader_status


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def select(self, _columns):
        return self

    def eq(self, _field, _value):
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


class _Supabase:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        assert name == "leagues_metadata"
        return _Query(self.rows)


def _meta_frame():
    return pd.DataFrame(
        [
            {
                "id": 1,
                "club_id": "club",
                "league_name": "Doubles League",
                "match_format": "doubles",
                "k_factor": 32,
                "status": "active",
                "is_active": True,
                "ended_at": None,
            },
            {
                "id": 2,
                "club_id": "club",
                "league_name": "Singles League",
                "match_format": "singles",
                "k_factor": 32,
                "status": "active",
                "is_active": True,
                "ended_at": None,
            },
        ]
    )


def test_active_league_expectations_enforce_match_format():
    singles = build_active_league_metadata_expectations(
        _meta_frame(),
        club_id="club",
        league_names={"Singles League"},
        default_k_factor=32,
        expected_match_format="singles",
    )
    assert singles[0]["expected"]["match_format"] == "singles"

    with pytest.raises(ValueError, match="doubles league"):
        build_active_league_metadata_expectations(
            _meta_frame(),
            club_id="club",
            league_names={"Doubles League"},
            default_k_factor=32,
            expected_match_format="singles",
        )


def test_match_uploader_status_separates_active_leagues(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "true")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES", "true")
    rows = [
        {"league_name": "Doubles League", "match_format": "doubles", "status": "active", "is_active": True, "ended_at": None},
        {"league_name": "Singles League", "match_format": "singles", "status": "active", "is_active": True, "ended_at": None},
        {"league_name": "Ended Singles", "match_format": "singles", "status": "ended", "is_active": False, "ended_at": "2026-01-01"},
    ]
    status = build_admin_match_uploader_status(_Supabase(rows), club_id="club")
    assert status["doubles_league_options"] == ["Doubles League"]
    assert status["singles_league_options"] == ["Singles League"]
    assert "Singles League" not in status["league_options"]


def test_source_contract_carries_league_format_everywhere():
    migration = Path("supabase/migrations/20260731033000_league_match_format.sql").read_text()
    singles = Path("jupr_app/domain/singles_match_processing.py").read_text()
    create = Path("jupr_app/services/admin_league_manager_create_service.py").read_text()
    routes = Path("services/api/admin_league_manager_routes.py").read_text()
    assert "add column if not exists match_format" in migration
    assert "expected_match_format=\"singles\"" in singles
    assert '"match_format": clean_match_format' in create
    assert 'pattern=r"^(doubles|singles)$"' in routes

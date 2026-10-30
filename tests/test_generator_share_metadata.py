from types import SimpleNamespace

import pytest

from jupr_app.services.generator_share_metadata_service import get_generator_share_metadata
from jupr_app.services.public_play_generator_service import PublicPlayGeneratorError


class Read:
    def __init__(self, row):
        self.row = row
        self.filters = {}

    def table(self, name):
        assert name == "live_sessions"
        return self

    def select(self, columns):
        # A projection in PostgREST must avoid transporting the entire JSON state.
        assert columns == "session_key,title,status,expires_at,mode:state->>mode"
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def limit(self, count):
        assert count == 1
        return self

    def execute(self):
        assert self.filters == {"club_id": "club", "session_key": "session"}
        return SimpleNamespace(data=[self.row] if self.row else [])


def test_share_metadata_is_scoped_and_contains_no_lineups_or_secrets():
    row = {"session_key": "session", "title": "Evening RR", "status": "active",
           "mode": "public_play_generator", "expires_at": "2099-01-01T00:00:00Z",
           "edit_token_hash": "must-not-leak", "state": {"private": True}}
    result = get_generator_share_metadata(Read(row), club_id="club", session_key="session")
    assert result == {"ok": True, "session": {"session_key": "session", "title": "Evening RR"}}


@pytest.mark.parametrize("row", [None, {"mode": "private"}, {
    "mode": "public_play_generator", "status": "active", "expires_at": "2000-01-01T00:00:00Z",
}])
def test_missing_private_and_expired_sessions_have_no_preview(row):
    with pytest.raises(PublicPlayGeneratorError):
        get_generator_share_metadata(Read(row), club_id="club", session_key="session")

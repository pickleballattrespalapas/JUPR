from __future__ import annotations

from dataclasses import dataclass

from jupr_app.ui.live.shared import LivePageConfig, _maybe_save_league_before_advance
from jupr_app.ui.pages.jupr_live import CLUB_SOCIAL_CONFIG, QUICK_SESSION_CONFIG


@dataclass
class _Ctx:
    admin_logged_in: bool = False


def test_quick_session_mode_stays_session_only_and_public_friendly():
    assert QUICK_SESSION_CONFIG.allow_official is False
    assert QUICK_SESSION_CONFIG.allow_tournament is False
    assert QUICK_SESSION_CONFIG.persistent_save_label is None
    assert QUICK_SESSION_CONFIG.event_types == ("Round Robin", "League / Ladder")


def test_club_social_mode_supports_rr_and_league_with_persistent_label():
    assert CLUB_SOCIAL_CONFIG.allow_official is False
    assert CLUB_SOCIAL_CONFIG.allow_tournament is False
    assert CLUB_SOCIAL_CONFIG.event_types == ("Round Robin", "League / Ladder")
    assert CLUB_SOCIAL_CONFIG.persistent_save_label == "Submit club social results"


def test_non_official_league_finalize_calls_callback_when_present():
    cfg = LivePageConfig(state_key="k", intro_markdown="x", allow_official=False)
    called = {"value": False}

    def _save(_ctx, _state, _event):
        called["value"] = True
        return True

    _maybe_save_league_before_advance(_Ctx(admin_logged_in=False), {}, {}, cfg, _save)
    assert called["value"] is True

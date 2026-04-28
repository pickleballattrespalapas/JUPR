from __future__ import annotations

from dataclasses import dataclass

from jupr_app.ui.live.shared import LivePageConfig, _maybe_save_league_before_advance
from jupr_app.ui.pages import jupr_live
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


class _FakeSt:
    def __init__(self):
        self.session_state = {jupr_live.SOCIAL_SKILL_LEVELS_KEY: ["All"]}
        self.errors: list[str] = []
        self.successes: list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def success(self, msg: str):
        self.successes.append(msg)


def test_club_social_save_handles_missing_tables_gracefully(monkeypatch):
    class _TablesMissing(jupr_live.SocialTablesNotInstalledError):
        pass

    fake_st = _FakeSt()

    monkeypatch.setattr(jupr_live, "st", fake_st)
    monkeypatch.setattr(
        jupr_live,
        "save_resolved_social_live_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(_TablesMissing("missing")),
    )

    ctx = type("Ctx", (), {"club_id": "club-1", "admin_logged_in": True})()
    ok = jupr_live._save_social(ctx, {}, {"type": "round_robin", "participants": []})

    assert ok is False
    assert fake_st.errors == [jupr_live.SOCIAL_TABLES_INSTALL_MESSAGE]
    assert not fake_st.successes

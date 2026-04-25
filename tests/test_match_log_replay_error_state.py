from __future__ import annotations

from types import SimpleNamespace

from jupr_app.ui.pages import match_log


def test_bulk_replay_error_state_persists_until_cleared(monkeypatch):
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(match_log, "st", fake_st)

    match_log._set_bulk_replay_error("Replay failed")
    assert fake_st.session_state[match_log.BULK_REPLAY_ERROR_STATE_KEY] == "Replay failed"

    match_log._clear_bulk_replay_error()
    assert match_log.BULK_REPLAY_ERROR_STATE_KEY not in fake_st.session_state

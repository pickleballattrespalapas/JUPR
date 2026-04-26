from pathlib import Path

from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS
from jupr_app.ui.pages.player_updates_subscribe import (
    requested_player_id_from_query,
    resolve_prefill_player_id,
)
import pandas as pd


def test_verified_updates_request_page_registered_as_hidden_public_route():
    assert "verified_updates_request" in PAGE_KEY_TO_LABEL
    assert PAGE_KEY_TO_LABEL["verified_updates_request"] == "📬 Verified Updates Request"
    assert "verified_updates_request" in HIDDEN_PAGE_KEYS
    assert "verified_updates_request" not in PUBLIC_NAV_KEYS


def test_players_page_uses_durable_verified_updates_route_params():
    contents = Path("jupr_app/ui/pages/players.py").read_text(encoding="utf-8")

    assert '"page": "verified_updates_request"' in contents
    assert '"player_id": int(pid)' in contents
    assert '"club_id": str(club_id)' in contents
    assert '[Subscribe to player updates]' in contents
    assert '"public": 1' not in contents


def test_verified_updates_route_maps_to_standalone_subscribe_page():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"📬 Verified Updates Request": player_updates_subscribe' in contents
    assert '"📬 Verified Updates Request": players' not in contents


def test_standalone_page_accepts_player_id_and_pid_query_params():
    assert requested_player_id_from_query(player_id_q="123", pid_q="999") == 123
    assert requested_player_id_from_query(player_id_q="", pid_q="999") == 999
    assert requested_player_id_from_query(player_id_q="", pid_q="") is None
    assert requested_player_id_from_query(player_id_q="not-a-number", pid_q="999") == 999


def test_invalid_or_missing_player_id_falls_back_to_universal_picker():
    options_df = pd.DataFrame([{"id": 10, "option_label": "Alice", "sort_label": "alice"}])
    assert resolve_prefill_player_id(options_df=options_df, player_id_q="", pid_q="") is None
    assert resolve_prefill_player_id(options_df=options_df, player_id_q="999", pid_q="") is None
    assert resolve_prefill_player_id(options_df=options_df, player_id_q="10", pid_q="") == 10


def test_standalone_subscribe_page_does_not_pop_player_query_params():
    contents = Path("jupr_app/ui/pages/player_updates_subscribe.py").read_text(encoding="utf-8")
    assert 'st.query_params.pop("player_id", None)' not in contents
    assert 'st.query_params.pop("pid", None)' not in contents


def test_standalone_subscribe_page_is_public_and_not_admin_gated():
    contents = Path("jupr_app/ui/pages/player_updates_subscribe.py").read_text(encoding="utf-8")
    assert "admin_logged_in" not in contents
    assert "admin_mode" not in contents


def test_streamlit_main_surfaces_page_render_exceptions_without_route_fallback():
    contents = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'Page render failed.' in contents
    assert 'st.error("This page failed to render, and navigation has been paused on this route.")' in contents
    assert 'st.exception(exc)' in contents
    assert 'Append ?debug=1 to the URL to view exception details in development.' in contents

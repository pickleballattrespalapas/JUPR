from pathlib import Path

from jupr_app.ui.page_registry import HIDDEN_PAGE_KEYS, PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS


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
    assert 'st.query_params["pid"] = str(pid)' in contents
    assert 'st.query_params["player_id"] = str(pid)' in contents
    assert 'st.query_params.pop("pid", None)' not in contents

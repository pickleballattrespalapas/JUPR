from jupr_app.ui.pages.session_console import (
    _derive_game_inputs,
    build_fuzzy_preview,
    build_session_route,
    parse_intake_names,
    parse_session_route,
    _is_round_complete,
    _round_has_unresolved_playoff,
    build_print_pack_html,
)


def test_build_session_route_variants():
    assert build_session_route("abc") == "sessions/abc"
    assert build_session_route("abc", round_number=2, court_id="3") == "sessions/abc/rounds/2/courts/3"


def test_parse_session_route_session_and_court():
    one = parse_session_route("sessions/abc")
    assert one == {
        "session_id": "abc",
        "round_number": None,
        "court_id": None,
        "is_court": False,
    }

    two = parse_session_route("/sessions/abc/rounds/2/courts/7/")
    assert two == {
        "session_id": "abc",
        "round_number": 2,
        "court_id": "7",
        "is_court": True,
    }


def test_parse_session_route_invalid_returns_none():
    assert parse_session_route("tournament/xyz") is None
    assert parse_session_route("") is None


def test_parse_intake_names_dedups_and_splits():
    parsed = parse_intake_names(" Alice\nBob, Alice ; Carla ")
    assert parsed == ["Alice", "Bob", "Carla"]


def test_build_fuzzy_preview_modes():
    existing = ["Alice Stone", "Bob Jones"]
    preview = build_fuzzy_preview(["Alice Stone", "Alic Ston", "Charlie"], existing)
    assert preview[0]["mode"] == "exact"
    assert preview[1]["mode"] == "fuzzy"
    assert preview[2]["mode"] == "new"


def test_derive_game_inputs_defaults_and_scored():
    assert _derive_game_inputs({"score_a": None, "score_b": None}) == ("teamA", 0)
    assert _derive_game_inputs({"score_a": 11, "score_b": 7}) == ("teamA", 7)
    assert _derive_game_inputs({"score_a": 5, "score_b": 11}) == ("teamB", 5)


def test_round_close_helpers_and_print_pack_html():
    court_items = [
        {"games": [{"score_a": 11, "score_b": 5}, {"score_a": 11, "score_b": 7}, {"score_a": 11, "score_b": 9}], "standings": [{"playoff_required": False}]},
        {"games": [{"score_a": 11, "score_b": 5}, {"score_a": 11, "score_b": 7}, {"score_a": 11, "score_b": 9}], "standings": [{"playoff_required": True}]},
    ]
    assert _is_round_complete(court_items, 4) is True
    assert _round_has_unresolved_playoff(court_items) is True

    html = build_print_pack_html(
        {"id": "s1"},
        {
            1: [
                {
                    "pod": {"court_number": 1},
                    "players": [{"player_order": 1, "player_id": 10}],
                    "games": [{"game_number": 1, "team_a_player_ids": [10, 11], "team_b_player_ids": [12, 13], "score_a": 11, "score_b": 6}],
                }
            ]
        },
    )
    assert "Session Print Pack" in html
    assert "Round 1 • Court 1" in html

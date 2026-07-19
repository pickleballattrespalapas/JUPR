from jupr_app.ui.pages import tournament_partner_board as partner_board
from jupr_app.ui.pages import tournament_registration as registration


def _choices():
    return [
        {
            "tournament": {"id": "111", "name": "Tournament A", "start_date": "2026-02-01"},
            "settings": {"registration_slug": "a-open"},
        },
        {
            "tournament": {"id": "222", "name": "Tournament B", "start_date": "2026-03-01"},
            "settings": {"registration_slug": "b-open"},
        },
    ]


def test_registration_resolve_public_tournament_id_prefers_valid_tournament_id():
    selected = registration._resolve_public_tournament_id(
        _choices(),
        qp_tournament_id="222",
        qp_slug="a-open",
    )
    assert selected == "222"


def test_registration_resolve_public_tournament_id_falls_back_to_slug_then_first():
    from_slug = registration._resolve_public_tournament_id(
        _choices(),
        qp_tournament_id="999",
        qp_slug="b-open",
    )
    assert from_slug == "222"

    from_first = registration._resolve_public_tournament_id(
        _choices(),
        qp_tournament_id="999",
        qp_slug="missing",
    )
    assert from_first == "111"


def test_partner_board_resolve_public_tournament_id_matches_rules():
    selected = partner_board.tournament_roster._resolve_public_tournament_id(
        _choices(),
        qp_tournament_id="",
        qp_slug="b-open",
    )
    assert selected == "222"

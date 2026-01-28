from jupr_app.domain.gamification.trophies import parse_tournament_podium_context


def test_parse_tournament_podium_context():
    tournament_id, placement = parse_tournament_podium_context("abc123:podium:2")

    assert tournament_id == "abc123"
    assert placement == 2


def test_parse_tournament_podium_context_fallbacks_to_value_num():
    tournament_id, placement = parse_tournament_podium_context("abc123:podium:2", value_num=3)

    assert tournament_id == "abc123"
    assert placement == 2

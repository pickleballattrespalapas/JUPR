from jupr_app.domain import ratings


def test_calculate_hybrid_elo_returns_zero_on_tie():
    d1, d2 = ratings.calculate_hybrid_elo(1200, 1200, 11, 11)
    assert d1 == 0.0
    assert d2 == 0.0


def test_calculate_hybrid_elo_winner_positive():
    d1, d2 = ratings.calculate_hybrid_elo(1200, 1200, 11, 7)
    assert d1 > 0
    assert d2 < 0 or d2 == 0

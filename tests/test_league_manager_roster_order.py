from jupr_app.ui.pages.league_manager import _roster_by_seed_strategy


def test_roster_by_seed_strategy_manual_keeps_uploaded_order():
    roster = [
        {"id": 1, "name": "B", "rating": 1300.0},
        {"id": 2, "name": "A", "rating": 1600.0},
    ]

    ordered = _roster_by_seed_strategy(roster, "manual")

    assert [p["id"] for p in ordered] == [1, 2]


def test_roster_by_seed_strategy_rating_sorts_high_to_low():
    roster = [
        {"id": 1, "name": "B", "rating": 1300.0},
        {"id": 2, "name": "A", "rating": 1600.0},
    ]

    ordered = _roster_by_seed_strategy(roster, "rating")

    assert [p["id"] for p in ordered] == [2, 1]

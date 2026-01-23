from jupr_app.domain.challenge_ladder import sorted_tiers


def test_sorted_tiers_orders_high_to_low():
    assert sorted_tiers(["DEV", "PREM", "EMER", "ADV"]) == ["PREM", "ADV", "DEV", "EMER"]

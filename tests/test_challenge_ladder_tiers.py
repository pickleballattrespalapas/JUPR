from jupr_app.domain.challenge_ladder import (
    ladder_can_initiate_challenge,
    ladder_can_receive_challenge,
    ladder_pair_eligibility,
    sorted_tiers,
)


def test_sorted_tiers_orders_high_to_low():
    assert sorted_tiers(["DEV", "PREM", "EMER", "ADV"]) == ["PREM", "ADV", "DEV", "EMER"]


def test_challenge_status_policy_matches_public_rulebook():
    assert ladder_can_initiate_challenge("Ready to Defend") is True
    assert ladder_can_initiate_challenge("Protected") is True
    assert ladder_can_initiate_challenge("Cooldown") is False
    assert ladder_can_receive_challenge("Ready to Defend") is True
    assert ladder_can_receive_challenge("Cooldown") is True
    assert ladder_can_receive_challenge("Protected") is False


def test_pair_eligibility_requires_tier_rank_range_and_status():
    eligible = ladder_pair_eligibility(
        challenger_tier="ADV",
        challenger_rank=6,
        challenger_status="Ready to Defend",
        defender_tier="ADV",
        defender_rank=2,
        defender_status="Cooldown",
        challenge_range=7,
    )
    blocked = ladder_pair_eligibility(
        challenger_tier="ADV",
        challenger_rank=10,
        challenger_status="Locked",
        defender_tier="INT",
        defender_rank=1,
        defender_status="Protected",
        challenge_range=7,
    )

    assert eligible == {"eligible": True, "reasons": [], "rank_gap": 4}
    assert blocked["eligible"] is False
    assert len(blocked["reasons"]) >= 3

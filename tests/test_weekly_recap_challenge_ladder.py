from jupr_app.domain.recaps.weekly_recap import _summarize_challenge_ladder_rows


def test_summarize_challenge_ladder_rows_groups_match_results_by_tier():
    rows = [
        {
            "id": 101,
            "tier_id": "PREM",
            "challenger_id": 10,
            "defender_id": 20,
            "winner_id": 10,
            "status": "COMPLETED",
            "challenger_rank_at_create": 8,
            "defender_rank_at_create": 4,
        },
        {
            "id": 102,
            "tier_id": "ADV",
            "challenger_id": 30,
            "defender_id": 40,
            "winner_id": 40,
            "status": "COMPLETED",
            "challenger_rank_at_create": 5,
            "defender_rank_at_create": 2,
        },
        {
            "id": 103,
            "tier_id": "ADV",
            "challenger_id": 50,
            "defender_id": 60,
            "winner_id": 50,
            "status": "PENDING_ACCEPTANCE",
            "challenger_rank_at_create": 1,
            "defender_rank_at_create": 3,
        },
    ]

    summary = _summarize_challenge_ladder_rows(
        rows,
        id_to_name={10: "Ana", 20: "Bob", 30: "Luis", 40: "Marco"},
    )

    assert summary == {
        "title": "Match Results",
        "by_tier": [
            {"tier": "Premier Tier", "lines": ["#8 Ana beat #4 Bob"]},
            {"tier": "Advanced Tier", "lines": ["#2 Marco defended vs #5 Luis"]},
        ],
    }


def test_summarize_challenge_ladder_rows_uses_unknown_rank_when_missing():
    rows = [
        {
            "id": 201,
            "tier_id": "EMER",
            "challenger_id": 99,
            "defender_id": 100,
            "winner_id": 100,
            "status": "COMPLETED",
            "challenger_rank_at_create": None,
            "defender_rank_at_create": "",
        }
    ]

    summary = _summarize_challenge_ladder_rows(rows, id_to_name={99: "Chal", 100: "Def"})

    assert summary == {
        "title": "Match Results",
        "by_tier": [
            {"tier": "Developing Tier", "lines": ["#? Def defended vs #? Chal"]},
        ],
    }


def test_summarize_challenge_ladder_rows_returns_empty_structure_for_no_completed_matches():
    rows = [
        {
            "id": 300,
            "tier_id": "PREM",
            "challenger_id": 1,
            "defender_id": 2,
            "winner_id": None,
            "status": "COMPLETED",
            "challenger_rank_at_create": 1,
            "defender_rank_at_create": 2,
        },
        {
            "id": 301,
            "tier_id": "ADV",
            "challenger_id": 3,
            "defender_id": 4,
            "winner_id": 4,
            "status": "FORFEITED",
            "challenger_rank_at_create": 3,
            "defender_rank_at_create": 4,
        },
    ]

    summary = _summarize_challenge_ladder_rows(rows, id_to_name={})

    assert summary == {"title": "Match Results", "by_tier": []}

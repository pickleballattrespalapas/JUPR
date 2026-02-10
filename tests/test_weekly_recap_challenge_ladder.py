from jupr_app.domain.recaps.weekly_recap import _summarize_challenge_ladder_rows


def test_summarize_challenge_ladder_rows_builds_counts_and_highlights():
    rows = [
        {
            "id": 101,
            "tier_id": "A1",
            "challenger_id": 10,
            "defender_id": 20,
            "winner_id": 10,
            "status": "COMPLETED",
        },
        {
            "id": 102,
            "tier_id": "A1",
            "challenger_id": 30,
            "defender_id": 40,
            "winner_id": 40,
            "status": "FORFEITED",
        },
        {
            "id": 103,
            "tier_id": "B2",
            "challenger_id": 50,
            "defender_id": 60,
            "winner_id": None,
            "status": "COMPLETED",
        },
    ]

    summary = _summarize_challenge_ladder_rows(
        rows,
        id_to_name={10: "Ada", 20: "Ben", 30: "Cal", 40: "Dee"},
    )

    assert summary["count"] == 2
    assert summary["by_tier"] == [{"tier_id": "A1", "count": 2}]
    assert summary["highlights"] == [
        {"display": "A1 • Ada beat Ben (Challenger win) • #101"},
        {"display": "A1 • Dee beat Cal (Defended) • #102"},
    ]


def test_summarize_challenge_ladder_rows_limits_highlights_to_six():
    rows = [
        {
            "id": idx,
            "tier_id": "T1",
            "challenger_id": idx,
            "defender_id": 999,
            "winner_id": idx,
            "status": "COMPLETED",
        }
        for idx in range(1, 9)
    ]

    summary = _summarize_challenge_ladder_rows(rows, id_to_name={})

    assert summary["count"] == 8
    assert len(summary["highlights"]) == 6

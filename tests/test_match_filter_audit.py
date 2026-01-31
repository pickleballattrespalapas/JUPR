import pandas as pd

from jupr_app.domain.match_filters import apply_match_filters_with_audit


def test_apply_match_filters_with_audit_tracks_removed_ids():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "club_id": "1",
                "date": "2024-02-10",
                "league": "Alpha",
                "match_type": "League",
                "is_valid": True,
                "context_type": "",
                "tournament_id": None,
                "is_void": False,
                "t1_p1": 1,
                "t1_p2": None,
                "t2_p1": 2,
                "t2_p2": None,
            },
            {"id": 2, "club_id": "2", "date": "2024-02-11", "match_type": "League"},
            {"id": 3, "club_id": "1", "date": "2024-02-12", "match_type": "PopUp"},
            {"id": 4, "club_id": "1", "date": "2024-02-13", "is_valid": False},
            {"id": 5, "club_id": "1", "date": "2024-02-14", "context_type": "TOURNAMENT"},
            {"id": 6, "club_id": "1", "date": "2024-02-15", "tournament_id": "99"},
            {"id": 7, "club_id": "1", "date": "2024-02-16", "match_type": "Tournament"},
            {"id": 8, "club_id": "1", "date": "2024-02-17", "is_void": True},
            {"id": 9, "club_id": "1", "date": "2024-01-05"},
            {
                "id": 10,
                "club_id": "1",
                "date": "2024-02-20",
                "t1_p1": 1,
                "t1_p2": None,
                "t2_p1": 3,
                "t2_p2": None,
            },
            {"id": 11, "club_id": "1", "date": "2024-03-05"},
        ]
    )

    filtered, audit = apply_match_filters_with_audit(
        df,
        {
            "club_id": "1",
            "exclude_popups": True,
            "start_date": "2024-02-01",
            "end_date": "2024-02-28",
            "eligible_player_ids": [1, 2],
        },
    )

    assert audit.raw_match_ids == [str(i) for i in range(1, 12)]
    assert audit.final_match_ids == ["1"]
    assert filtered["id"].astype(str).tolist() == ["1"]

    removed_by_step = {step.step_name: step.removed_match_ids for step in audit.steps}
    assert removed_by_step["club_id"] == ["2"]
    assert removed_by_step["exclude_popups"] == ["3"]
    assert removed_by_step["is_valid"] == ["4"]
    assert removed_by_step["exclude_context_type_tournament"] == ["5"]
    assert removed_by_step["tournament_id_is_null"] == ["6"]
    assert removed_by_step["exclude_match_type_tournament"] == ["7"]
    assert removed_by_step["exclude_is_void"] == ["8"]
    assert removed_by_step["start_date"] == ["9"]
    assert removed_by_step["end_date"] == ["11"]
    assert removed_by_step["eligible_player_ids"] == ["10"]

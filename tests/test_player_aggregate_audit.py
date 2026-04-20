from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.player_aggregate_audit import compute_player_aggregate_reconciliation


def _ctx(df_players: pd.DataFrame, df_matches: pd.DataFrame):
    return SimpleNamespace(df_players_all=df_players, df_matches=df_matches, club_id="club")


def test_reconciliation_detects_players_aggregate_larger_than_canonical_wins():
    ctx = _ctx(
        pd.DataFrame([{"id": 1, "wins": 112, "losses": 46, "matches_played": 158}]),
        pd.DataFrame(
            [
                {
                    "id": "m1",
                    "club_id": "club",
                    "date": "2024-01-01",
                    "score_t1": 11,
                    "score_t2": 5,
                    "t1_p1": 1,
                    "t2_p1": 2,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
                {
                    "id": "m2",
                    "club_id": "club",
                    "date": "2024-01-02",
                    "score_t1": 11,
                    "score_t2": 9,
                    "t1_p1": 1,
                    "t2_p1": 3,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
                {
                    "id": "m3",
                    "club_id": "club",
                    "date": "2024-01-03",
                    "score_t1": 11,
                    "score_t2": 7,
                    "t1_p1": 1,
                    "t2_p1": 4,
                    "match_type": "PopUp",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
            ]
        ),
    )

    result = compute_player_aggregate_reconciliation(ctx, player_id=1, club_id="club")

    assert result["players_table_wins"] == 112
    assert result["filtered_match_distinct_win_match_ids"] == 2
    assert result["wins_delta"] == 110
    assert result["popup_match_count_for_player"] == 1
    assert result["aggregate_out_of_sync_warning"] is True


def test_reconciliation_exact_alignment_case():
    ctx = _ctx(
        pd.DataFrame([{"id": 7, "wins": 1, "losses": 1, "matches_played": 2}]),
        pd.DataFrame(
            [
                {
                    "id": "m1",
                    "club_id": "club",
                    "date": "2024-01-01",
                    "score_t1": 11,
                    "score_t2": 9,
                    "t1_p1": 7,
                    "t2_p1": 2,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
                {
                    "id": "m2",
                    "club_id": "club",
                    "date": "2024-01-02",
                    "score_t1": 7,
                    "score_t2": 11,
                    "t1_p1": 7,
                    "t2_p1": 3,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
            ]
        ),
    )

    result = compute_player_aggregate_reconciliation(ctx, player_id=7, club_id="club")

    assert result["wins_delta"] == 0
    assert result["losses_delta"] == 0
    assert result["matches_delta"] == 0
    assert result["aggregate_out_of_sync_warning"] is False


def test_reconciliation_uses_distinct_win_match_ids_for_high_roller_semantics():
    ctx = _ctx(
        pd.DataFrame([{"id": 11, "wins": 2, "losses": 1, "matches_played": 3}]),
        pd.DataFrame(
            [
                {
                    "id": "m1",
                    "club_id": "club",
                    "date": "2024-01-01",
                    "score_t1": 11,
                    "score_t2": 8,
                    "t1_p1": 11,
                    "t1_p2": 99,
                    "t2_p1": 2,
                    "t2_p2": 3,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
                {
                    "id": "m1",
                    "club_id": "club",
                    "date": "2024-01-01",
                    "score_t1": 11,
                    "score_t2": 8,
                    "t1_p1": 11,
                    "t1_p2": 99,
                    "t2_p1": 2,
                    "t2_p2": 3,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
                {
                    "id": "m2",
                    "club_id": "club",
                    "date": "2024-01-02",
                    "score_t1": 5,
                    "score_t2": 11,
                    "t1_p1": 11,
                    "t2_p1": 4,
                    "match_type": "League",
                    "is_valid": True,
                    "context_type": "LEAGUE",
                    "tournament_id": None,
                },
            ]
        ),
    )

    result = compute_player_aggregate_reconciliation(ctx, player_id=11, club_id="club")

    assert result["filtered_match_win_rows"] == 2
    assert result["filtered_match_distinct_win_match_ids"] == 1
    assert result["filtered_duplicate_match_rows_for_player"] == 2

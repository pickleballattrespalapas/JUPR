from jupr_app.domain.tournaments.bracket_builder import SUPPORTED_TEAM_COUNTS, build_round_robin_games


def test_supported_team_counts_include_six() -> None:
    assert SUPPORTED_TEAM_COUNTS == list(range(4, 17))


def test_round_robin_6_team_template_loads_from_csv() -> None:
    team_ids_by_number = {team_number: f"team-{team_number}" for team_number in range(1, 7)}

    games = build_round_robin_games(tournament_id="tour-1", team_ids_by_number=team_ids_by_number)

    assert len(games) == 15
    assert {game["rr_round_number"] for game in games} == {1, 2, 3, 4, 5}
    assert {game["rr_slot_number"] for game in games} == {1, 2, 3}

    pairings = {
        tuple(sorted((game["team_a_id"], game["team_b_id"])))
        for game in games
    }
    assert len(pairings) == 15


def test_large_round_robins_schedule_every_pair_once_without_round_conflicts() -> None:
    for team_count in (9, 16):
        team_ids = {
            team_number: f"team-{team_number}"
            for team_number in range(1, team_count + 1)
        }
        games = build_round_robin_games(
            tournament_id="tour-large",
            team_ids_by_number=team_ids,
        )

        assert len(games) == team_count * (team_count - 1) // 2
        assert len(
            {
                tuple(sorted((game["team_a_id"], game["team_b_id"])))
                for game in games
            }
        ) == len(games)
        for round_number in {game["rr_round_number"] for game in games}:
            appearances = [
                team_id
                for game in games
                if game["rr_round_number"] == round_number
                for team_id in (game["team_a_id"], game["team_b_id"])
            ]
            assert len(appearances) == len(set(appearances))

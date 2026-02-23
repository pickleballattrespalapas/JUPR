from jupr_app.domain.session_ladder_engine import (
    applyMovement,
    computeCourtStandings,
    generateRoundGames,
    getMovers,
    resolveTies,
)


def test_generate_round_games_4p_template():
    games = generateRoundGames([1, 2, 3, 4], "4p")
    assert len(games) == 3
    assert games[0]["teamA"] == [1, 2] and games[0]["teamB"] == [3, 4]
    assert games[1]["teamA"] == [1, 3] and games[1]["teamB"] == [2, 4]
    assert games[2]["teamA"] == [1, 4] and games[2]["teamB"] == [2, 3]


def test_generate_round_games_5p_template_with_rotation():
    games = generateRoundGames([10, 11, 12, 13, 14], "5p")
    assert len(games) == 5
    assert [g["sit_out"] for g in games] == [10, 11, 12, 13, 14]


def test_compute_standings_stats_correctness():
    players = [1, 2, 3, 4]
    games = [
        {"teamA": [1, 2], "teamB": [3, 4], "scoreA": 21, "scoreB": 10},
        {"teamA": [1, 3], "teamB": [2, 4], "scoreA": 15, "scoreB": 21},
        {"teamA": [1, 4], "teamB": [2, 3], "scoreA": 21, "scoreB": 18},
    ]
    standings = computeCourtStandings(games, players)
    by_id = {row["player_id"]: row for row in standings}

    assert by_id[1]["wins"] == 2
    assert by_id[1]["losses"] == 1
    assert by_id[1]["pf"] == 57
    assert by_id[1]["pa"] == 49
    assert by_id[1]["pd"] == 8


def test_resolve_ties_uses_head_to_head_before_playoff_required():
    standings = [
        {"player_id": 1, "wins": 2, "losses": 1, "pf": 50, "pa": 40, "pd": 10, "rank": 1},
        {"player_id": 2, "wins": 2, "losses": 1, "pf": 50, "pa": 40, "pd": 10, "rank": 2},
        {"player_id": 3, "wins": 1, "losses": 2, "pf": 40, "pa": 50, "pd": -10, "rank": 3},
        {"player_id": 4, "wins": 1, "losses": 2, "pf": 40, "pa": 50, "pd": -10, "rank": 4},
    ]
    games = [
        {"teamA": [1, 3], "teamB": [2, 4], "scoreA": 10, "scoreB": 21},
        {"teamA": [1, 4], "teamB": [2, 3], "scoreA": 21, "scoreB": 10},
        {"teamA": [1, 3], "teamB": [2, 4], "scoreA": 9, "scoreB": 21},
    ]
    resolved = resolveTies(standings, games)

    assert resolved[0]["player_id"] == 2
    assert resolved[0]["tie_break"] == "HeadToHead"
    assert resolved[0]["playoff_required"] is False


def test_resolve_ties_marks_playoff_required_when_h2h_still_tied():
    players = [1, 2, 3, 4]
    # players 1 and 2 tie in wins/pd/pf and never oppose each other directly
    games = [
        {"teamA": [1, 2], "teamB": [3, 4], "scoreA": 21, "scoreB": 10},
        {"teamA": [1, 2], "teamB": [3, 4], "scoreA": 10, "scoreB": 21},
    ]
    standings = computeCourtStandings(games, players)
    resolved = resolveTies(standings, games)
    group = [row for row in resolved if row["player_id"] in {1, 2}]

    assert all(row["playoff_required"] is True for row in group)
    assert all(row["tie_break"] == "PlayoffRequired" for row in group)


def _build_10_court_rankings(players_per_court: int = 4) -> list[list[int]]:
    pods = []
    base = 1
    for _ in range(10):
        pods.append(list(range(base, base + players_per_court)))
        base += players_per_court
    return pods


def test_apply_movement_10_courts_one_mover():
    pods = _build_10_court_rankings(4)
    moved = applyMovement(pods, 1)

    assert moved[0] == [1, 2, 3, 5]
    assert moved[1] == [4, 6, 7, 9]
    assert moved[8] == [32, 34, 35, 37]
    assert moved[9] == [36, 38, 39, 40]


def test_apply_movement_10_courts_two_movers_and_get_movers():
    pods = _build_10_court_rankings(5)
    moved = applyMovement(pods, 2)

    assert moved[0] == [1, 2, 3, 6, 7]
    assert moved[1] == [4, 5, 8, 11, 12]
    assert moved[9] == [44, 45, 48, 49, 50]

    standings = [{"player_id": pid, "rank": idx + 1} for idx, pid in enumerate([100, 101, 102, 103, 104])]
    movers = getMovers(standings, 2)
    assert movers == {"up": [100, 101], "down": [103, 104]}

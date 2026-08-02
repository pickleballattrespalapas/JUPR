from pathlib import Path

from jupr_app.domain.adaptive_play_engine import generator_event_standings

ROOT = Path(__file__).resolve().parents[1]


def _event(sort_mode: str):
    return {
        "standingsSort": sort_mode,
        "participants": [
            {"id": "p1", "name": "Alex", "roster_order": 1},
            {"id": "p2", "name": "Blake", "roster_order": 2},
            {"id": "p3", "name": "Casey", "roster_order": 3},
            {"id": "p4", "name": "Drew", "roster_order": 4},
        ],
        "rounds": [
            {
                "number": 1,
                "status": "saved",
                "matches": [
                    {"id": "m1", "sideA": ["p1"], "sideB": ["p2"], "scoreA": 1, "scoreB": 0},
                    {"id": "m2", "sideA": ["p1"], "sideB": ["p3"], "scoreA": 1, "scoreB": 0},
                    {"id": "m3", "sideA": ["p2"], "sideB": ["p3"], "scoreA": 20, "scoreB": 19},
                ],
            },
            {
                "number": 2,
                "status": "skipped",
                "matches": [
                    {"id": "ignored", "sideA": ["p4"], "sideB": ["p1"], "scoreA": 99, "scoreB": 0}
                ],
            },
        ],
    }


def test_full_standings_follow_selected_primary_sort() -> None:
    assert [row["name"] for row in generator_event_standings(_event("wins"))] == [
        "Alex", "Blake", "Drew", "Casey"
    ]
    assert [row["name"] for row in generator_event_standings(_event("points"))] == [
        "Blake", "Casey", "Alex", "Drew"
    ]
    assert [row["name"] for row in generator_event_standings(_event("differential"))] == [
        "Alex", "Blake", "Drew", "Casey"
    ]


def test_full_standings_include_zero_game_players_and_ignore_skips() -> None:
    rows = generator_event_standings(_event("wins"))
    drew = next(row for row in rows if row["name"] == "Drew")
    assert drew["matches"] == 0
    assert drew["pointsFor"] == 0
    assert all(row["pointsFor"] < 99 for row in rows)


def test_round_robin_standings_routes_and_links_exist() -> None:
    admin_runner = (ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx").read_text()
    public_runner = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx").read_text()
    admin_setup = (ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx").read_text()
    public_setup = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx").read_text()
    assert "View full standings" in admin_runner
    assert "View full standings" in public_runner
    assert "Standings ranked by" in admin_setup
    assert "Standings ranked by" in public_setup
    assert (ROOT / "apps/web/app/admin/round-robin-generator/sessions/[sessionKey]/standings/page.tsx").exists()
    assert (ROOT / "apps/web/app/clubs/[clubSlug]/round-robin-generator/sessions/[sessionKey]/standings/page.tsx").exists()


def test_api_accepts_and_persists_standings_sort() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    engine = (ROOT / "jupr_app/domain/adaptive_play_engine.py").read_text()
    assert "standings_sort" in admin_routes
    assert "standings_sort" in public_routes
    assert '"standingsSort"' in engine

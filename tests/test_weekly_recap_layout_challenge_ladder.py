from jupr_app.ui.components.weekly_recap_layout import build_weekly_recap_html


def _base_recap(challenge_ladder: dict) -> dict:
    return {
        "week_start": "2026-01-05",
        "week_end": "2026-01-11",
        "numbers": {},
        "spotlight": [],
        "around_club": {"leagues": [], "round_robins": []},
        "around_descriptions": {},
        "challenge_ladder": challenge_ladder,
        "looking_ahead": [],
        "meta": {},
    }


def test_build_weekly_recap_html_renders_challenge_ladder_match_results_by_tier():
    recap = _base_recap(
        {
            "title": "Match Results",
            "by_tier": [
                {"tier": "Premier Tier", "lines": ["#8 Ana beat #4 Bob"]},
                {"tier": "Advanced Tier", "lines": ["#2 Marco defended vs #5 Luis"]},
            ],
        }
    )

    html = build_weekly_recap_html(recap, print_view=False)

    assert "Challenge Ladder" in html
    assert "<div class='muted-label'>Match Results</div>" in html
    assert "<div class='muted-label'>Premier Tier:</div>" in html
    assert "#8 Ana beat #4 Bob" in html
    assert "<div class='muted-label'>Advanced Tier:</div>" in html
    assert "#2 Marco defended vs #5 Luis" in html


def test_build_weekly_recap_html_hides_challenge_ladder_without_lines():
    recap = _base_recap({"title": "Match Results", "by_tier": [{"tier": "Premier Tier", "lines": []}]})

    html = build_weekly_recap_html(recap, print_view=False)

    assert "Challenge Ladder" not in html

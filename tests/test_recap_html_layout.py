from jupr_app.ui.components.weekly_recap_layout import build_recap_html, build_weekly_recap_html


def _base_recap() -> dict:
    return {
        "week_start": "2026-02-10",
        "week_end": "2026-02-16",
        "numbers": {},
        "spotlight": [],
        "around_club": {"leagues": [], "round_robins": []},
        "around_descriptions": {},
        "challenge_ladder": {},
        "looking_ahead": [],
        "meta": {},
    }


def test_featured_past_event_section_rendered_when_present() -> None:
    recap = _base_recap()
    recap["featured_past_event"] = {
        "title": "Winter Open Finals",
        "datetime": "2026-02-12T19:00:00-06:00",
        "location": "Center Court",
        "summary_bullets": ["Packed crowd"],
    }

    html = build_weekly_recap_html(recap, print_view=False)

    assert "id='featured-past-event'" in html
    assert "Winter Open Finals" in html


def test_tournaments_section_rendered_with_subheadings() -> None:
    recap = _base_recap()
    recap["tournaments_in_period"] = [
        {"title": "Baja Classic", "datetime": "2026-02-14T09:00:00-06:00", "location": "Court A"}
    ]
    recap["upcoming_tournaments"] = [
        {
            "title": "Spring Open",
            "datetime": "2026-02-21T09:00:00-06:00",
            "location": "Court B",
            "reg_url": "https://example.com/register",
        }
    ]

    html = build_weekly_recap_html(recap, print_view=False)

    assert "id='tournaments'" in html
    assert "Tournaments This Period" in html
    assert "Upcoming Tournaments" in html
    assert "Register" in html


def test_monthly_member_of_month_anchor_rendered_when_provided() -> None:
    recap = _base_recap()
    recap["member_of_month"] = {"name": "Ana", "summary": "Great leadership."}

    html = build_recap_html(recap, level="monthly", print_view=False)

    assert "id='member-of-month'" in html
    assert "Ana" in html

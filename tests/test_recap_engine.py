from datetime import datetime

from jupr_app.domain.recaps.engine import (
    compute_recap,
    is_tournament_item,
    validate_featured_past_event,
)


def test_tournament_classifier_prefers_type_over_keyword_fallback() -> None:
    typed_non_tournament = {
        "title": "Summer Tournament Preview",
        "event_type": "social",
        "category": None,
        "source": "events",
    }
    typed_tournament = {
        "title": "Weekly Mixer",
        "event_type": "tournament",
        "category": None,
        "source": "events",
    }
    keyword_fallback = {
        "title": "Baja Open",
        "event_type": None,
        "category": None,
        "source": "events",
    }

    assert is_tournament_item(typed_non_tournament) is False
    assert is_tournament_item(typed_tournament) is True
    assert is_tournament_item(keyword_fallback) is True


def test_validate_featured_past_event_in_window() -> None:
    report_start = datetime.fromisoformat("2026-02-10T00:00:00-06:00")
    report_end = datetime.fromisoformat("2026-02-17T00:00:00-06:00")

    ok_result = validate_featured_past_event(
        {"datetime": "2026-02-12T20:00:00-06:00"},
        report_start,
        report_end,
    )
    bad_result = validate_featured_past_event(
        {"datetime": "2026-02-17T00:00:00-06:00"},
        report_start,
        report_end,
    )

    assert ok_result.ok is True
    assert bad_result.ok is False


def test_compute_recap_buckets_events_and_tournaments() -> None:
    report_start = datetime.fromisoformat("2026-02-10T00:00:00-06:00")
    report_end = datetime.fromisoformat("2026-02-17T00:00:00-06:00")
    now = datetime.fromisoformat("2026-02-17T00:00:00-06:00")

    events_rows = [
        {
            "id": "event-1",
            "club_id": "club-1",
            "name": "Members Night",
            "event_type": "social",
            "starts_at": "2026-02-12T18:00:00-06:00",
        },
        {
            "id": "event-2",
            "club_id": "club-1",
            "name": "Club Open",
            "starts_at": "2026-02-14T09:00:00-06:00",
        },
        {
            "id": "event-3",
            "club_id": "club-1",
            "name": "Sunset Social",
            "event_type": "social",
            "starts_at": "2026-02-20T18:00:00-06:00",
        },
    ]
    tournaments_rows = [
        {
            "id": "tour-1",
            "club_id": "club-1",
            "name": "Winter Cup",
            "created_at": "2026-02-11T10:00:00-06:00",
        },
        {
            "id": "tour-2",
            "club_id": "club-1",
            "name": "Spring Classic",
            "created_at": "2026-02-19T11:00:00-06:00",
        },
    ]

    recap = compute_recap(
        "club-1",
        report_start,
        report_end,
        lookahead_days=7,
        now=now,
        featured_past_event={"event_id": "event-2", "link_label": "Recap"},
        events_rows=events_rows,
        tournaments_rows=tournaments_rows,
    )

    assert [item["event_id"] for item in recap["events_in_period"]] == ["event-1"]
    assert [item["event_id"] for item in recap["tournaments_in_period"]] == ["tour-1", "event-2"]
    assert [item["event_id"] for item in recap["upcoming_events"]] == ["event-3"]
    assert [item["event_id"] for item in recap["upcoming_tournaments"]] == ["tour-2"]
    assert recap["featured_past_event"]["title"] == "Club Open"
    assert recap["featured_past_event"]["link_label"] == "Recap"

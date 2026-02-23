from jupr_app.domain.recaps.store import (
    FeaturedPastEvent,
    FeaturedUpcomingEvent,
    InMemoryRecapStore,
    RecapRecord,
)


def _base_recap() -> RecapRecord:
    return RecapRecord(
        recap_id="recap-1",
        club_id="club-1",
        level="weekly",
        status="draft",
        report_start="2026-02-09T00:00:00-06:00",
        report_end="2026-02-16T00:00:00-06:00",
        featured_upcoming_event=FeaturedUpcomingEvent(
            event_id="event-upcoming-1",
            title="Club Mixer",
            datetime="2026-02-20T18:00:00-06:00",
            location="Center Court",
            reg_url="https://example.com/register",
            pitch="Sign up now!",
        ),
    )


def test_featured_past_event_can_be_stored() -> None:
    store = InMemoryRecapStore()
    recap = _base_recap()
    recap.featured_past_event = FeaturedPastEvent(
        event_id="event-past-1",
        title="Winter Doubles",
        datetime="2026-02-10T19:00:00-06:00",
        location="Court 2",
        summary_bullets=["Great turnout", "Tight finals"],
        link_url="https://example.com/results",
        link_label="Full Results",
    )

    store.save(recap)
    saved = store.get(recap.recap_id)

    assert saved is not None
    assert saved.featured_past_event is not None
    assert saved.featured_past_event.event_id == "event-past-1"


def test_content_snapshot_can_store_tournaments_lists() -> None:
    store = InMemoryRecapStore()
    recap = _base_recap()
    recap.content_snapshot = {
        "tournaments_in_period": [
            {
                "title": "Winter Open",
                "datetime": "2026-02-12T10:00:00-06:00",
                "location": "Main Courts",
                "results_link": "https://example.com/winter-open",
                "winners": ["A Team", "B Team"],
            }
        ],
        "upcoming_tournaments": [
            {
                "title": "Spring Classic",
                "datetime": "2026-02-28T09:00:00-06:00",
                "location": "North Courts",
                "reg_url": "https://example.com/spring-classic",
            }
        ],
    }

    store.save(recap)
    saved = store.get(recap.recap_id)

    assert saved is not None
    assert saved.content_snapshot["tournaments_in_period"][0]["title"] == "Winter Open"
    assert saved.content_snapshot["upcoming_tournaments"][0]["title"] == "Spring Classic"


def test_publish_defaults_visibility_to_public() -> None:
    store = InMemoryRecapStore()
    recap = _base_recap()
    recap.visibility = "private"
    store.save(recap)

    published = store.publish(recap.recap_id)

    assert published.status == "published"
    assert published.visibility == "public"

from jupr_app.domain.recaps.weekly_recap import _resolve_event_key


def test_match_with_tournament_id_without_context_type_is_tournament():
    match = {"tournament_id": "tour-1", "match_type": "League", "league": "Alpha"}
    assert _resolve_event_key(match) is None


def test_match_with_tournament_week_tag_is_tournament():
    match = {"week_tag": "Tournament", "match_type": "League", "league": "Alpha"}
    assert _resolve_event_key(match) is None


def test_popup_with_tournament_id_is_tournament_not_rr_even_if_popup_flagged():
    match = {
        "is_popup": True,
        "match_type": "PopUp",
        "tournament_id": "tour-2",
        "context_type": None,
        "league": "PopUp",
        "week_tag": "Friday",
    }
    assert _resolve_event_key(match) is None


def test_context_event_match_is_classified_as_popup_rr():
    match = {
        "match_type": "League",
        "context_type": "event",
        "context_id": "event-123",
        "league": "Alpha",
    }
    assert _resolve_event_key(match) == ("RR", "event-123")

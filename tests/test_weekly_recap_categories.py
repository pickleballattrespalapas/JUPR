from jupr_app.domain.recaps import weekly_recap


def _stats_row(wins: int, games: int, delta: float, losses: int = 0) -> dict:
    return {
        "wins": wins,
        "losses": losses,
        "games": games,
        "delta_jupr": delta,
    }


def test_event_highlights_default_featured_limit_is_three():
    stats = {
        1: _stats_row(6, 8, 0.20, losses=2),
        2: _stats_row(5, 9, 0.15, losses=4),
        3: _stats_row(4, 7, 0.10, losses=3),
        4: _stats_row(3, 6, 0.05, losses=3),
    }

    highlights = weekly_recap._event_highlights(stats, {1: "A", 2: "B", 3: "C", 4: "D"}, count=1, short_labels=False, prefer_jump=False)

    assert len(highlights) == 1
    assert highlights[0]["key"] == "TOP_PERFORMER"
    assert [player["id"] for player in highlights[0]["players"]] == [1, 2, 3]


def test_event_highlights_respects_per_category_max_featured(monkeypatch):
    original = weekly_recap.RECAP_CATEGORY_CONFIG
    monkeypatch.setattr(
        weekly_recap,
        "RECAP_CATEGORY_CONFIG",
        {
            **original,
            "TOP_PERFORMER": {
                "label": "Top Performer",
                "max_featured": 2,
            },
            "BIGGEST_JUMP": {
                "label": "Biggest Jump",
                "max_featured": 1,
            },
        },
    )
    stats = {
        1: _stats_row(6, 8, 0.20, losses=2),
        2: _stats_row(5, 9, 0.15, losses=4),
        3: _stats_row(4, 7, 0.10, losses=3),
    }

    highlights = weekly_recap._event_highlights(stats, {1: "A", 2: "B", 3: "C"}, count=2, short_labels=False, prefer_jump=True)

    assert highlights[0]["key"] == "BIGGEST_JUMP"
    assert len(highlights[0]["players"]) == 1
    assert len(highlights[1]["players"]) == 2


def test_event_highlights_handles_empty_categories_without_errors():
    stats = {1: _stats_row(3, 5, 0, losses=2)}

    highlights = weekly_recap._event_highlights(stats, {1: "A"}, count=2, short_labels=False, prefer_jump=True)

    assert len(highlights) == 1
    assert highlights[0]["key"] == "TOP_PERFORMER"
    assert highlights[0]["players"]

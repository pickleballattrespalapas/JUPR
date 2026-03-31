from jupr_app.ui.pages.weekly_recap_admin import _normalize_overrides


def test_normalize_overrides_preserves_spotlight_keys_and_ordering():
    generated = [
        {"key": "TOP_PERFORMER_WEEK", "candidate_ids": ["a"], "description": "top", "order": 1, "include": True},
        {"key": "COMMUNITY_STANDOUT_WEEK", "candidate_ids": ["b"], "description": "community", "order": 3, "include": True},
        {"key": "SOCIAL_GRIND_WEEK", "candidate_ids": ["c"], "description": "grind", "order": 4, "include": True},
    ]
    overrides = {
        "COMMUNITY_STANDOUT_WEEK": {"players": ["b2"], "description": "edited", "order": 2, "include": True},
    }

    normalized = _normalize_overrides(overrides, generated)

    assert "COMMUNITY_STANDOUT_WEEK" in normalized
    assert "SOCIAL_GRIND_WEEK" in normalized
    assert normalized["COMMUNITY_STANDOUT_WEEK"]["players"] == ["b2"]
    assert normalized["COMMUNITY_STANDOUT_WEEK"]["order"] == 2
    assert normalized["SOCIAL_GRIND_WEEK"]["players"] == ["c"]

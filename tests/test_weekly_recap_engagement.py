from copy import deepcopy

import pytest

from jupr_app.domain.recaps.engagement import POLLS, clean_engagement, new_engagement
from jupr_app.domain.recaps.recap_builder import public_edition_report
from tests.test_admin_match_log_service import FakeSupabase


def test_automatic_question_uses_selected_topic_and_avoids_recent_club_prompts():
    db = FakeSupabase({"club_recap_preferences": [{"club_id": "one", "settings": {"engagement_mode": "review"}}],
                       "recap_engagement_editions": []})
    report = {"report": {"edition": [{"items": [{"title": "Clinic for resets and drops"}]}]}}
    first = new_engagement(db, "one", "2026-10-26", report)
    assert first["poll"]["id"] == "drops-resets" and first["mode"] == "review"
    db.tables["recap_engagement_editions"].append({"club_id": "one", "week_start": "2026-10-26", "scope_key": "Whole club", "config": first})
    assert new_engagement(db, "one", "2026-11-02", report)["poll"]["id"] != first["poll"]["id"]
    assert new_engagement(db, "two", "2026-11-02", report)["poll"]["id"] == first["poll"]["id"]


def test_public_report_only_announces_availability_and_rejects_ambiguous_choices():
    raw = {"mode": "automatic", "featured": "poll", "poll": deepcopy(POLLS[0]), "body": "private email wording"}
    public = public_edition_report({"engagement": clean_engagement(raw), "edition": []})
    assert public["has_engagement"] and "engagement" not in public and "private email wording" not in str(public)
    raw["poll"]["options"] = ["Drops", " drops "]
    with pytest.raises(ValueError, match="different answers"):
        clean_engagement(raw)

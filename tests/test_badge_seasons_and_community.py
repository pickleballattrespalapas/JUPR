from dataclasses import asdict
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from jupr_app.domain.gamification.award_identity import award_key
from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_registry import awardable_badge_ids, registry
from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.community_awards import build_community_award
from jupr_app.domain.gamification.evaluators import (
    evaluate_battle_tested, evaluate_consistency, evaluate_mr_reliable,
    evaluate_good_sport, evaluate_community_builder, evaluate_mentor,
)
from jupr_app.domain.gamification.seasons import BadgeSeason, season_match_groups, validate_badge_seasons


def season(**overrides):
    return dict(id="winter", club_id="club", name="Winter play", start_date="2025-11-01",
                end_date="2026-03-31", timezone="UTC", **overrides)


def fact(i, date_text="2026-01-01T12:00:00Z", *, player_id=1, win=True, club_id="club"):
    return dict(player_id=player_id, match_id=f"m{i:03d}", date_dt=pd.Timestamp(date_text),
                club_id=club_id, win=win, season_key="2026")


def context(rows, *, seasons=None, cutoff="2026-04-01T00:00:00Z"):
    raw = SimpleNamespace(badge_seasons=seasons if seasons is not None else [season()])
    return BadgeEvaluationContext("club", None, pd.Timestamp(cutoff), raw, pd.DataFrame(rows), pd.DataFrame())


def test_seasons_use_admin_dates_include_last_day_and_never_invent_a_calendar_year():
    rows = [fact(1, "2025-11-01T00:00:00Z"), fact(2, "2026-03-31T23:59:59Z"),
            fact(3, "2025-10-31T23:59:59Z"), fact(4, "2026-04-01T00:00:00Z"),
            fact(5, club_id="another-club")]
    configured, pid, matches = list(season_match_groups(context(rows)))[0]
    assert pid == 1
    assert matches["match_id"].tolist() == ["m001", "m002"]
    assert configured.context_id == "badge-season:winter"
    assert list(season_match_groups(context(rows, seasons=[]))) == []
    delattr((missing := context(rows)).ctx, "badge_seasons")
    assert list(season_match_groups(missing)) == []


def test_season_dates_respect_local_midnight_and_dst():
    configured = BadgeSeason.from_row(dict(season(), start_date="2026-03-08", end_date="2026-03-08", timezone="America/Los_Angeles"))
    assert configured.start == pd.Timestamp("2026-03-08T08:00:00Z")
    assert configured.end_exclusive == pd.Timestamp("2026-03-09T07:00:00Z")


def test_seasons_reject_overlap_and_invalid_ranges_and_ignore_other_clubs():
    with pytest.raises(ValueError, match="overlap"):
        validate_badge_seasons([season(), dict(season(), id="spring", start_date="2026-03-31")], club_id="club")
    with pytest.raises(ValueError, match="end date"):
        validate_badge_seasons([dict(season(), start_date="2026-05-01")], club_id="club")
    with pytest.raises(ValueError, match="unique"):
        validate_badge_seasons([season(), season()], club_id="club")
    assert len(validate_badge_seasons([season(), dict(season(), club_id="another-club")], club_id="club")) == 1
    adjacent = dict(season(), id="spring", start_date="2026-04-01", end_date="2026-05-31")
    assert len(validate_badge_seasons([season(), adjacent], club_id="club")) == 2


def test_battle_tested_requires_50_distinct_matches_in_one_configured_season():
    rows = [fact(i) for i in range(49)] + [fact(0), fact(90, "2025-10-31T12:00:00Z"), fact(91, player_id=2)]
    assert evaluate_battle_tested(context(rows)) == []
    rows += [fact(49), fact(50)]
    award, = evaluate_battle_tested(context(rows[::-1]))
    assert award.match_id == "m049"
    assert award.value_json["matches"] == 50
    assert len(award.value_json["qualifying_match_ids"]) == 50
    later = dict(season(), id="summer", start_date="2026-04-01", end_date="2026-08-31")
    rows += [fact(i + 100, "2026-05-01T12:00:00Z") for i in range(50)]
    awards = evaluate_battle_tested(context(rows, seasons=[season(), later], cutoff="2026-06-01T00:00:00Z"))
    assert {a.context_id for a in awards} == {"badge-season:winter", "badge-season:summer"}


def test_consistency_crosses_iso_week_53_but_requires_six_consecutive_weeks_in_one_season():
    start = pd.Timestamp("2020-12-07T12:00:00Z")
    rows = [fact(i, (start + pd.Timedelta(weeks=i)).isoformat()) for i in range(8)]
    winter = dict(season(), start_date="2020-11-01", end_date="2021-03-31")
    award, = evaluate_consistency(context(rows[::-1], seasons=[winter], cutoff="2021-04-01T00:00:00Z"))
    assert award.match_id == "m005"
    assert award.value_json["weeks"] == ["2020-W50", "2020-W51", "2020-W52", "2020-W53", "2021-W01", "2021-W02"]
    assert evaluate_consistency(context(rows[:5], seasons=[winter], cutoff="2021-04-01T00:00:00Z")) == []
    missing_week = [row for row in rows if row["match_id"] != "m003"]
    assert evaluate_consistency(context(missing_week, seasons=[winter], cutoff="2021-04-01T00:00:00Z")) == []
    shorter = dict(winter, start_date="2020-12-28")
    assert evaluate_consistency(context(rows, seasons=[shorter], cutoff="2021-04-01T00:00:00Z")) == []


def test_mr_reliable_waits_for_end_and_uses_final_record_including_late_losses():
    rows = [fact(i, win=i < 21) for i in range(30)]
    assert evaluate_mr_reliable(context(rows, cutoff="2026-03-31T23:59:59Z")) == []
    award, = evaluate_mr_reliable(context(rows + [fact(0)]))
    assert award.value_json["win_pct"] == 0.7
    assert award.value_json["matches"] == 30
    assert award.match_id is None
    assert evaluate_mr_reliable(context(rows + [fact(31, "2026-03-31T23:59:59Z", win=False)])) == []
    assert evaluate_mr_reliable(context(rows[:29])) == []
    assert evaluate_mr_reliable(context(rows + [fact(31, "2026-04-01T00:00:00Z", win=False)])) == [award]


def recognition(**overrides):
    args = dict(club_id="club", player_id=1, badge_id="good_sport", recognition_id=str(uuid4()),
                criteria=["honest_calls"], note="Corrected an out call in the final game.", contribution_date=date(2026, 9, 6))
    return build_community_award(**(args | overrides))


def test_repeat_community_awards_have_separate_identity_and_retries_keep_same_identity():
    first = recognition()
    retry = recognition(recognition_id=first.value_json["recognition_id"])
    another = recognition()
    assert first == retry
    assert award_key(asdict(first)) != award_key(asdict(another))
    assert first.value_json["qualifying_actions"] == ["Honest calls, even when they cost a point"]
    assert first.context_type == "overall"
    assert first.match_id is None


@pytest.mark.parametrize("override", [
    {"badge_id": "first_win"}, {"criteria": []}, {"criteria": "honest_calls"},
    {"criteria": ["volunteer"]}, {"note": " "}, {"recognition_id": "not-an-id"}, {"player_id": True},
])
def test_community_award_requires_a_qualifying_action_and_valid_recognition(override):
    with pytest.raises(ValueError):
        recognition(**override)


def test_community_badges_are_repeatable_and_never_automatically_inferred():
    definitions = {b.badge_id: b for b in BADGE_DEFINITIONS}
    for badge_id, evaluator in [("good_sport", evaluate_good_sport), ("community_builder", evaluate_community_builder), ("mentor", evaluate_mentor)]:
        assert evaluator(context([fact(i) for i in range(200)])) == []
        assert definitions[badge_id].is_stackable
        assert registry()[badge_id].metric_source_policy == "non_match"
        assert badge_id not in awardable_badge_ids()

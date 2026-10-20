from datetime import date
import json

import pandas as pd
import pytest

from jupr_app.domain.recaps.weekly_recap import compute_weekly_recap, get_spotlight_candidates, _filter_matches
from jupr_app.domain.recaps.recap_evidence import metric_candidates
from tests.test_weekly_recap_range_driven import _ctx
from tests.test_weekly_report import match


@pytest.mark.parametrize('reverse', [False, True])
def test_mixed_iso_precision_preserves_games_and_metric_evidence(reverse):
    dates = ['2026-10-16T15:01:43.5174+00:00', '2026-10-17T19:00:00+00:00',
             '2026-10-18T12:00:00Z']
    games = [match(i, date=value) for i, value in enumerate(dates, 1)]
    if reverse:
        games.reverse()
    frame = pd.DataFrame(games)
    assert _filter_matches(frame, include_tournaments=True).date_dt.notna().all()
    ctx = _ctx(frame)
    report = compute_weekly_recap(ctx, date(2026, 10, 14), date(2026, 10, 20))
    candidates = get_spotlight_candidates(ctx, date(2026, 10, 14), date(2026, 10, 20))
    assert report['numbers']['matches'] == 3
    assert candidates['METRIC_most_wins'][0]['value_json']['games'] == 3
    assert candidates['METRIC_playing_days'][0]['value_json']['value'] == 3
    json.dumps({'report': report, 'candidates': candidates}, allow_nan=False)


def test_legacy_date_only_and_timestamp_local_day_bounds_are_preserved():
    games = [match(1, date='2026-10-18'),
             match(2, date='2026-10-19T05:00:00.123Z'),
             match(3, date='2026-10-19T08:00:00Z')]
    report = compute_weekly_recap(_ctx(pd.DataFrame(games)), date(2026, 10, 18), date(2026, 10, 18))
    assert report['numbers']['matches'] == 2


def test_invalid_metric_timestamp_fails_explicitly_without_partial_metrics():
    frame = _filter_matches(pd.DataFrame([match(1, date='invalid')]), include_tournaments=True)
    with pytest.raises(ValueError, match='valid match timestamp'):
        metric_candidates(frame, {1: 'A', 2: 'B', 3: 'C', 4: 'D'}, 'America/Mazatlan')

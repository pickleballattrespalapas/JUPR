from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.award_identity import award_key
from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import evaluate_clutch_performer


def context(rows, day=31):
    return BadgeEvaluationContext(
        'club', None, datetime(2026, 1, day, tzinfo=timezone.utc),
        SimpleNamespace(), pd.DataFrame(rows), pd.DataFrame(),
    )


def match(i, *, win=True, margin=2, player_id=1):
    return dict(player_id=player_id, match_id=str(i), win=win, margin=margin,
                date_dt=pd.Timestamp(f'2026-01-{i:02d}', tz='UTC'))


def test_clutch_requires_five_distinct_close_wins_and_ignores_losses_and_invalid_scores():
    rows = [match(i) for i in range(1, 5)]
    rows += [match(1), match(6, win=False, margin=-1), match(7, margin=3), match(8, margin=0)]
    assert evaluate_clutch_performer(context(rows)) == []
    rows += [match(9, margin=1)]
    award, = evaluate_clutch_performer(context(rows[::-1]))
    assert award.player_id == 1
    assert award.match_id == '9'
    assert award.value_json['qualifying_match_ids'] == ['1', '2', '3', '4', '9']


def test_clutch_uses_first_fifth_win_and_respects_cutoff_and_player_boundaries():
    rows = [match(i) for i in range(1, 7)] + [match(i, player_id=2) for i in range(1, 5)]
    assert evaluate_clutch_performer(context(rows, day=4)) == []
    award, = evaluate_clutch_performer(context(rows))
    assert (award.player_id, award.match_id) == (1, '5')


def test_clutch_recognizes_existing_and_revoked_legacy_lifetime_keys():
    base = dict(player_id=1, badge_id='clutch_performer')
    legacy = dict(base, context_type='overall', context_id=None, revoked_at='2026-02-01')
    new = dict(base, context_type='overall', context_id='clutch_performer:lifetime')
    assert award_key(legacy) == award_key(new)

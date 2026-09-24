from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import evaluate_above_expectations, evaluate_breakthrough
from jupr_app.domain.gamification.rivalries import rivalry_candidates


def context(wins, cutoff=None):
    facts = pd.DataFrame([dict(player_id=1,match_id=str(i),date_dt=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=i),
                               opponent_ids=[2,3,3],win=won) for i,won in enumerate(wins)])
    return BadgeEvaluationContext('club',None,cutoff,SimpleNamespace(),facts,pd.DataFrame())


def test_rivalry_is_identified_before_followup_wins_and_remains_tracked_after_recovery():
    ctx=context([True,True,False,False,False,False,True,True,True,True])
    found=rivalry_candidates(ctx,'nemesis_found')
    assert len(found)==1 and found[0].match_id=='5'
    wins=rivalry_candidates(ctx,'rivalry_win')
    assert len(wins)==8
    assert {a.match_id for a in wins}=={'6','7','8','9'}
    assert {a.value_json['opponent_id'] for a in wins}=={2,3}
    streaks=rivalry_candidates(ctx,'rivalry_streak')
    assert len(streaks)==2 and {a.match_id for a in streaks}=={'8'}
    settled=rivalry_candidates(ctx,'settled_the_score')
    assert len(settled)==2 and {a.match_id for a in settled}=={'7'}


def test_rivalry_respects_cutoff_duplicate_facts_and_separate_streaks():
    ctx=context([False]*6+[True]*3+[False]+[True]*3)
    before=rivalry_candidates(ctx,'rivalry_streak')
    assert {a.match_id for a in before}=={'8','12'}
    ctx.facts=pd.concat([ctx.facts,ctx.facts]).sort_values('date_dt',ascending=False)
    assert [asdict(a) for a in rivalry_candidates(ctx,'rivalry_streak')]==[asdict(a) for a in before]
    ctx.as_of=pd.Timestamp('2026-01-06',tz='UTC')
    assert rivalry_candidates(ctx,'rivalry_win')==[]
    assert len(rivalry_candidates(ctx,'nemesis_found'))==1


def test_above_expectations_requires_known_probability_and_known_available_delta():
    ctx=context([True,True,True])
    ctx.facts['league']='A'
    ctx.facts['margin']=4
    ctx.facts['expected_win_prob']=[float('nan'),0.4,0.4]
    ctx.facts['abs_elo_delta']=[10,float('nan'),10]
    awards=evaluate_above_expectations(ctx)
    assert [a.match_id for a in awards]==['2']


def test_breakthrough_never_invents_rank_movement_from_ties_or_input_order():
    ctx=context([])
    ctx.ctx.df_leagues=pd.DataFrame([dict(player_id=i,league_name='A',rating=1200,starting_rating=1200,matches_played=20) for i in range(30)])
    assert evaluate_breakthrough(ctx)==[]
    ctx.ctx.df_leagues=ctx.ctx.df_leagues.sample(frac=1,random_state=7)
    assert evaluate_breakthrough(ctx)==[]

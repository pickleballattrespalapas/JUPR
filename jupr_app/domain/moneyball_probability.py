"""Versioned Moneyball probability and pre-event calibration engine."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_config(event):
    players = [{k:p[k] for k in ('id','name','rating')} for p in event['players']]
    matches = [{k:m[k] for k in ('id','round','team1','team2')} for m in event['matches']]
    return {'name':event['name'],'players':players,'matches':matches}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_config(config):
    players, matches = config['players'], config['matches']
    if len(players) != 8 or len(matches) != 14:
        raise ValueError('This prototype requires exactly 8 players and 14 doubles games.')
    ids = [p['id'] for p in players]
    if len(set(ids)) != 8:
        raise ValueError('Player IDs must be unique.')
    for p in players:
        if not isinstance(p['name'],str) or not p['name'].strip() or not np.isfinite(p['rating']) or not 1 <= p['rating'] <= 8:
            raise ValueError('Each player needs a name and a finite decimal rating between 1 and 8.')
    game_count = dict.fromkeys(ids,0)
    partners, opponents = {}, {}
    match_ids = set()
    for m in matches:
        if m['id'] in match_ids:
            raise ValueError('Match IDs must be unique.')
        match_ids.add(m['id'])
        t1,t2 = m['team1'],m['team2']
        if len(t1) != 2 or len(t2) != 2 or len(set(t1+t2)) != 4 or not set(t1+t2).issubset(ids):
            raise ValueError('Each doubles match needs four different, known players.')
        for pid in t1+t2:
            game_count[pid] += 1
        for t in (t1,t2):
            pair = tuple(sorted(t))
            partners[pair] = partners.get(pair,0)+1
        for a in t1:
            for b in t2:
                pair = tuple(sorted((a,b)))
                opponents[pair] = opponents.get(pair,0)+1
    if set(game_count.values()) != {7} or len(partners) != 28 or set(partners.values()) != {1} or len(opponents) != 28 or set(opponents.values()) != {2}:
        raise ValueError('Use a complete perfect RR: everyone plays 7 games, partners once and opposes twice.')


def match_probabilities(model, differences):
    support = np.array(model['margin_support'],dtype=int)
    mag = np.abs(support)-2
    a = np.array(model['log_magnitude_weights'])[mag]
    logits = a[None,:]+model['tilt_coefficient']*np.asarray(differences)[:,None]*support[None,:]
    logits -= logits.max(axis=1)[:,None]
    p = np.exp(logits)
    p /= p.sum(axis=1)[:,None]
    return p


def distributions(config,model):
    validate_config(config)
    ids = [p['id'] for p in config['players']]
    index = {pid:i for i,pid in enumerate(ids)}
    ratings = {p['id']:p['rating'] for p in config['players']}
    a = np.zeros((14,8),dtype=np.int16)
    differences = []
    for j,m in enumerate(config['matches']):
        differences.append(sum(ratings[pid] for pid in m['team1'])/2-sum(ratings[pid] for pid in m['team2'])/2)
        for pid in m['team1']:
            a[j,index[pid]] = 1
        for pid in m['team2']:
            a[j,index[pid]] = -1
    support = np.array(model['margin_support'],dtype=np.int16)
    probabilities = match_probabilities(model,differences)
    game_means = probabilities@support
    expected_pd = game_means@a
    percentiles, pmfs = [], []
    for player in range(8):
        pmf = np.array([1.])
        for j in np.flatnonzero(a[:,player]):
            single = np.zeros(23)
            single[support+11] = probabilities[j]
            if a[j,player] < 0:
                single = single[::-1]
            pmf = np.convolve(pmf,single)
        assert len(pmf) == 155 and abs(pmf.sum()-1) < 1e-12
        cdf = np.cumsum(pmf)
        # Midpoint CDF avoids counting a result's whole discrete probability mass as beaten.
        percentiles.append(cdf-.5*pmf)
        pmfs.append(pmf)
    return a,support,probabilities,np.array(percentiles),np.array(pmfs),expected_pd


def simulate(a,support,probabilities,percentiles,number,seed):
    rng = np.random.default_rng(seed)
    margins = np.empty((number,14),dtype=np.int16)
    for j,p in enumerate(probabilities):
        cdf = np.cumsum(p)
        cdf[-1] = 1
        margins[:,j] = support[np.searchsorted(cdf,rng.random(number),side='right')]
    totals = margins@a
    wins = ((margins > 0).astype(np.int16)@np.maximum(a,0)+(margins < 0).astype(np.int16)@np.maximum(-a,0))
    u = np.empty((number,8))
    for i in range(8):
        u[:,i] = percentiles[i,totals[:,i]+77]
    z = np.log(np.clip(u,1e-12,1-1e-12))-np.log1p(-np.clip(u,1e-12,1-1e-12))
    return totals,wins,u,z


def winning_shares(scores):
    best = scores.max(axis=1)[:,None]
    tied = np.isclose(scores,best,rtol=0,atol=1e-12)
    return np.mean(tied/tied.sum(axis=1)[:,None],axis=0)


def balance(z,tolerance=.0005,max_iterations=70):
    offsets = np.zeros(8)
    history = []
    for iteration in range(max_iterations):
        chances = winning_shares(z+offsets)
        error = float(np.max(np.abs(chances-.125)))
        history.append({'iteration':iteration,'max_first_place_deviation':error})
        if error <= tolerance:
            break
        # Inputs are simulated before scoring; adjustments never look at played results.
        gain = 3.0 if iteration < 30 else 1.5
        offsets += gain*(.125-chances)
        offsets -= offsets.mean()
    return offsets,history


def public_chances(players,chances,simulations):
    return [{'player_id':p['id'],'player':p['name'],'first_place_chance_percent':float(chances[i]*100),
             'approximate_95_percent_monte_carlo_error_points':float(1.96*np.sqrt(chances[i]*(1-chances[i])/simulations)*100)} for i,p in enumerate(players)]


def prepare(event,model,calibration_sims=200000,audit_sims=300000,seed=2026101402):
    if calibration_sims < 10000 or audit_sims < 10000:
        raise ValueError('Use at least 10,000 calibration and audit simulations.')
    config = canonical_config(event)
    a,support,p,pct,pmfs,expected_pd = distributions(config,model)
    totals,wins,u,z = simulate(a,support,p,pct,calibration_sims,seed)
    offsets,history = balance(z)
    base_chances = winning_shares(z)
    train_chances = winning_shares(z+offsets)
    del totals, wins, u, z
    # Independent second batch checks fairness; this batch never tunes offsets.
    test_totals,test_wins,test_u,test_z = simulate(a,support,p,pct,audit_sims,seed+1)
    audit_chances = winning_shares(test_z+offsets)
    points_chances = winning_shares(test_totals-expected_pd)
    examples = []
    score_matrix = test_z+offsets
    winner = score_matrix.argmax(axis=1)
    for i,pinfo in enumerate(config['players']):
        eligible = np.flatnonzero((winner == i)&(test_wins[:,i] <= 3))
        example_index = int(eligible[0]) if len(eligible) else int(np.flatnonzero(winner == i)[0])
        examples.append({'player_id':pinfo['id'],'player':pinfo['name'],'simulated_winning_record':f"{test_wins[example_index,i]}-{7-test_wins[example_index,i]}",
                         'simulated_point_differential':int(test_totals[example_index,i]),'simulation_row':example_index,
                         'label':'Simulated illustration, not a played game or promised result.'})
    max_error = float(np.max(np.abs(audit_chances-.125)))
    lock = {
        'version':'fair-moneyball-event-v1','config':config,'config_sha256':digest(config),'model_sha256':digest(model),'model':model,
        'seed':seed,'margin_probabilities':p.tolist(),'player_total_margin_pmf':pmfs.tolist(),
        'player_percentiles':pct.tolist(),'balance_logit_offsets':offsets.tolist(),'expected_point_differentials':expected_pd.tolist(),
        'entry_fee_mxn':100,'prizes_mxn':[800],
        'tie_policy':'Exact score ties share the total prizes for the positions occupied.',
        'fairness':{
            'target_first_place_chance_percent':12.5,'calibration_simulations':calibration_sims,'audit_simulations':audit_sims,
            'unbalanced_percentile_chances':public_chances(config['players'],base_chances,calibration_sims),
            'calibration_chances':public_chances(config['players'],train_chances,calibration_sims),
            'independent_audit_chances':public_chances(config['players'],audit_chances,audit_sims),
            'raw_points_above_expectation_chances':public_chances(config['players'],points_chances,audit_sims),
            'max_audit_deviation_percentage_points':max_error*100,
            'simulation_fairness_check_passed':bool(max_error <= .005),
            'adjustment_iterations':len(history),'optimization_history':history,
            'simulated_winning_examples':examples,
            'claim_scope':'Winning chances under this score model with accurate frozen ratings and conditional match independence; not a real-world guarantee.',
        },
    }
    lock['lock_sha256'] = digest(lock)
    return lock


def score(event,lock):
    verified = dict(lock)
    supplied_hash = verified.pop('lock_sha256')
    if digest(verified) != supplied_hash:
        raise ValueError('Frozen setup was changed; prepare a new event before play.')
    if not lock['fairness']['simulation_fairness_check_passed']:
        raise ValueError('This setup did not pass its simulation fairness check; do not use it as a balanced event.')
    if digest(canonical_config(event)) != lock['config_sha256']:
        raise ValueError('Players, ratings or schedule changed after setup was frozen.')
    ids = [p['id'] for p in lock['config']['players']]
    index = {pid:i for i,pid in enumerate(ids)}
    totals = np.zeros(8,dtype=int)
    wins = np.zeros(8,dtype=int)
    for m in event['matches']:
        s1,s2 = m.get('score1'),m.get('score2')
        if isinstance(s1,bool) or isinstance(s2,bool) or not isinstance(s1,int) or not isinstance(s2,int) or min(s1,s2) < 0 or abs(s1-s2) < 2 or max(s1,s2) < 11 or (max(s1,s2) > 11 and abs(s1-s2) != 2):
            raise ValueError('All games need completed scores to 11, win by two; extended games must finish two points apart.')
        margin = s1-s2
        if abs(margin)>11:
            raise ValueError('A finished game to 11 cannot have a margin above 11.')
        for team,sign in [(m['team1'],1),(m['team2'],-1)]:
            for pid in team:
                i=index[pid]
                totals[i] += sign*margin
                wins[i] += int(sign*margin>0)
    rows=[]
    percentiles=np.array(lock['player_percentiles'])
    game_probabilities=np.array(lock['margin_probabilities'])
    support=np.array(lock['model']['margin_support'])
    team1_win_probability=game_probabilities[:,support>0].sum(axis=1)
    expected_wins=np.zeros(8)
    for j,m in enumerate(lock['config']['matches']):
        for pid in m['team1']:
            expected_wins[index[pid]]+=team1_win_probability[j]
        for pid in m['team2']:
            expected_wins[index[pid]]+=1-team1_win_probability[j]
    for i,p in enumerate(lock['config']['players']):
        u=float(percentiles[i,totals[i]+77])
        u=float(np.clip(u,1e-12,1-1e-12))
        z=np.log(u)-np.log1p(-u)+lock['balance_logit_offsets'][i]
        fair_score=float(100/(1+np.exp(-z)))
        rows.append({'player_id':p['id'],'player':p['name'],'wins':int(wins[i]),'losses':int(7-wins[i]),
                     'expected_wins':float(expected_wins[i]),
                     'actual_point_differential':int(totals[i]),
                     'expected_point_differential':float(lock['expected_point_differentials'][i]),
                     'performance_percentile':u*100,'moneyball_score':fair_score,
                     'simulated_first_place_chance_percent':lock['fairness']['independent_audit_chances'][i]['first_place_chance_percent']})
    rows.sort(key=lambda r:r['moneyball_score'],reverse=True)
    j=0
    prizes=lock['prizes_mxn']+[0]*(8-len(lock['prizes_mxn']))
    while j<8:
        end=j+1
        while end<8 and abs(rows[end]['moneyball_score']-rows[j]['moneyball_score']) < 1e-10:
            end+=1
        prize=sum(prizes[j:end])/(end-j)
        for row in rows[j:end]:
            row['place']=j+1
            row['prize_mxn']=round(prize,2)
            row['net_after_entry_mxn']=round(prize-lock['entry_fee_mxn'],2)
        j=end
    # Any cent rounding is allocated in rank order, preserving the fixed pool.
    drift=round(sum(lock['prizes_mxn'])-sum(r['prize_mxn'] for r in rows),2)
    if drift:
        rows[0]['prize_mxn']=round(rows[0]['prize_mxn']+drift,2)
        rows[0]['net_after_entry_mxn']=round(rows[0]['prize_mxn']-lock['entry_fee_mxn'],2)
    return {'event_name':event['name'],'method':'Fair Play Moneyball v1','entry_fee_mxn':100,'prize_pool_mxn':800,
            'standings':rows,'fairness':lock['fairness'],'holdout_validation':lock['model']['validation'],
            'setup_lock_sha256':lock['lock_sha256'],'experimental':True,'individual_ratings_displayed':False}


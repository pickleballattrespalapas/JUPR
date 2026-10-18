"""Versioned Moneyball probability and pre-event calibration engine."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_config(event):
    players = [{k:p[k] for k in ('id','name','rating')} for p in event['players']]
    matches = [{k:m[k] for k in ('id','round','team1','team2')} for m in event['matches']]
    config = {'name':event['name'],'players':players,'matches':matches}
    # Preserve the hashes of existing 11-point locks.
    if event.get('goal_points', 11) != 11:
        config['goal_points'] = event['goal_points']
    return config


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_config(config):
    if type(config.get('goal_points', 11)) is not int or config.get('goal_points', 11) not in {11, 15}:
        raise ValueError('Moneyball supports games to 11 or 15, win by two.')
    players, matches = config['players'], config['matches']
    count = len(players)
    six_player = count == 6
    if six_player:
        if len(matches) != 6 or sorted(m['round'] for m in matches) != list(range(1, 7)):
            raise ValueError('Six-player Moneyball requires six rounds with one game per round.')
    elif not 4 <= count <= 40 or count % 4 not in {0, 1} or len(matches) != count*(count-1)//4:
        raise ValueError('Use six players or a supported complete perfect doubles round robin.')
    ids = [p['id'] for p in players]
    if len(set(ids)) != count:
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
    if six_player:
        if (set(game_count.values()) != {4} or len(partners) != 12 or set(partners.values()) != {1}
                or len(opponents) != 15 or set(opponents.values()) != {1, 2}):
            raise ValueError('Six-player Moneyball requires four games and four distinct partners each, facing every opponent once or twice.')
        for pid in ids:
            byes = sorted(m['round'] for m in matches if pid not in m['team1'] + m['team2'])
            if not (len(byes) == 2 and byes[0] <= 3 < byes[1] and byes[1] - byes[0] > 1):
                raise ValueError('Six-player Moneyball requires one bye in each three-round block and no consecutive byes.')
    elif set(game_count.values()) != {count-1} or len(partners) != count*(count-1)//2 or set(partners.values()) != {1} or len(opponents) != count*(count-1)//2 or set(opponents.values()) != {2}:
        raise ValueError('Use a complete perfect RR: everyone plays every partner once, partners once and opposes twice.')


def scoring_point_margins(goal, point_probability):
    """Exact win-by-two margins for independent scoring points, including deuce.

    This models awarded points, not service exchanges or rally scoring rules.
    """
    q = np.asarray(point_probability, dtype=float)
    if q.ndim != 1 or np.any(~np.isfinite(q)) or np.any((q <= 0) | (q >= 1)):
        raise ValueError('Scoring-point probabilities must be strictly between zero and one.')
    support = np.array(list(range(-goal, -1)) + list(range(2, goal + 1)))
    p = np.empty((len(q), len(support)))
    deuce = math.comb(2*goal-2, goal-1) * (q*(1-q))**(goal-1)
    for j, margin in enumerate(support):
        winner = q if margin > 0 else 1-q
        loser_points = goal-abs(int(margin))
        p[:, j] = math.comb(goal+loser_points-1, loser_points) * winner**goal * (1-winner)**loser_points
        if abs(margin) == 2:
            p[:, j] += deuce * winner**2 / (q*q+(1-q)**2)
    return p / p.sum(axis=1)[:, None]


def model_for_target(base_model, goal):
    if type(goal) is not int or goal not in {11, 15}:
        raise ValueError('Moneyball supports games to 11 or 15, win by two.')
    if goal == 11:
        return base_model
    return {
        'version': 'fair-moneyball-15-point-extension-v1',
        'kind': 'scoring_point_extension', 'goal_points': 15, 'win_by_two': True,
        'margin_support': list(range(-15, -1)) + list(range(2, 16)),
        'source_model': base_model,
        'validation': {'historical_15_point_validation': False,
                       'interpretation': 'Experimental extension of the 11-point model; not fitted or validated on 15-point results.'},
        'assumptions': [
            'Independent awarded scoring points conditional on frozen team ratings.',
            'Per-point probability is calibrated to the original model\'s 11-point win probability, then used for games to 15.',
            'No outcomes from this event are used to prepare expectations or balance offsets.',
        ],
    }


def match_probabilities(model, differences):
    if model.get('kind') == 'scoring_point_extension':
        source = model['source_model']
        reference = match_probabilities(source, differences)
        wins = reference[:, np.array(source['margin_support']) > 0].sum(axis=1)
        low, high = np.full(len(wins), 1e-12), np.full(len(wins), 1-1e-12)
        source_support = np.array(source['margin_support'])
        for _ in range(48):
            mid = (low+high)/2
            predicted = scoring_point_margins(source['goal_points'], mid)[:, source_support > 0].sum(axis=1)
            low = np.where(predicted < wins, mid, low)
            high = np.where(predicted >= wins, mid, high)
        return scoring_point_margins(model['goal_points'], (low+high)/2)
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
    a = np.zeros((len(config['matches']),len(ids)),dtype=np.int16)
    differences = []
    for j,m in enumerate(config['matches']):
        differences.append(sum(ratings[pid] for pid in m['team1'])/2-sum(ratings[pid] for pid in m['team2'])/2)
        for pid in m['team1']:
            a[j,index[pid]] = 1
        for pid in m['team2']:
            a[j,index[pid]] = -1
    support = np.array(model['margin_support'],dtype=np.int16)
    bound = int(np.max(np.abs(support)))
    probabilities = match_probabilities(model,differences)
    game_means = probabilities@support
    expected_pd = game_means@a
    percentiles, pmfs = [], []
    for player in range(len(ids)):
        pmf = np.array([1.])
        for j in np.flatnonzero(a[:,player]):
            single = np.zeros(2*bound+1)
            single[support+bound] = probabilities[j]
            if a[j,player] < 0:
                single = single[::-1]
            pmf = np.convolve(pmf,single)
        assert len(pmf) == 2*bound*np.count_nonzero(a[:,player])+1 and abs(pmf.sum()-1) < 1e-12
        cdf = np.cumsum(pmf)
        # Midpoint CDF avoids counting a result's whole discrete probability mass as beaten.
        percentiles.append(cdf-.5*pmf)
        pmfs.append(pmf)
    return a,support,probabilities,np.array(percentiles),np.array(pmfs),expected_pd


def simulate(a,support,probabilities,percentiles,number,seed):
    rng = seed if isinstance(seed,np.random.Generator) else np.random.default_rng(seed)
    count = a.shape[1]
    totals = np.zeros((number,count),dtype=np.int16)
    wins = np.zeros((number,count),dtype=np.int16)
    for j,p in enumerate(probabilities):
        cdf = np.cumsum(p)
        cdf[-1] = 1
        margin = support[np.searchsorted(cdf,rng.random(number),side='right')]
        for i in np.flatnonzero(a[j]):
            signed = margin*a[j,i]
            totals[:,i] += signed
            wins[:,i] += signed > 0
    u = np.empty((number,count))
    for i in range(count):
        u[:,i] = percentiles[i,totals[:,i]+int(np.max(np.abs(support)))*np.count_nonzero(a[:,i])]
    clipped = np.clip(u,1e-12,1-1e-12)
    z = np.log(clipped)-np.log1p(-clipped)
    return totals,wins,u,z


def winning_shares(scores):
    best = scores.max(axis=1)[:,None]
    tied = np.isclose(scores,best,rtol=0,atol=1e-12)
    return np.mean(tied/tied.sum(axis=1)[:,None],axis=0)


def balance(z,tolerance=.001,max_iterations=100):
    count = z.shape[1]
    target = 1/count
    offsets = np.zeros(count)
    history = []
    for iteration in range(max_iterations):
        chances = winning_shares(z+offsets)
        error = float(np.max(np.abs(chances-target)))
        history.append({'iteration':iteration,'max_first_place_deviation':error})
        if error <= tolerance:
            break
        # Inputs are simulated before scoring; adjustments never look at played results.
        gain = (3.0 if iteration < 50 else 1.5)*count/8
        offsets += gain*(target-chances)
        offsets -= offsets.mean()
    return offsets,history


def public_chances(players,chances,simulations):
    return [{'player_id':p['id'],'player':p['name'],'first_place_chance_percent':float(chances[i]*100),
             'approximate_95_percent_monte_carlo_error_points':float(1.96*np.sqrt(chances[i]*(1-chances[i])/simulations)*100)} for i,p in enumerate(players)]


def prepare(event,model,calibration_sims=200000,audit_sims=300000,seed=2026101402):
    if calibration_sims < 10000 or audit_sims < 10000:
        raise ValueError('Use at least 10,000 calibration and audit simulations.')
    config = canonical_config(event)
    if config.get('goal_points', 11) != model['goal_points']:
        raise ValueError('The scoring model must match the event target score.')
    a,support,p,pct,pmfs,expected_pd = distributions(config,model)
    count = len(config['players'])
    batch_size = 20000
    rng = np.random.default_rng(seed)
    z = np.empty((calibration_sims,count))
    for start in range(0,calibration_sims,batch_size):
        size = min(batch_size,calibration_sims-start)
        z[start:start+size] = simulate(a,support,p,pct,size,rng)[3]
    offsets,history = balance(z)
    base_chances = winning_shares(z)
    train_chances = winning_shares(z+offsets)
    del z
    audit_chances = np.zeros(count)
    points_chances = np.zeros(count)
    rng = np.random.default_rng(seed+1)
    for start in range(0,audit_sims,batch_size):
        size = min(batch_size,audit_sims-start)
        totals,_,_,z = simulate(a,support,p,pct,size,rng)
        audit_chances += winning_shares(z+offsets)*size/audit_sims
        points_chances += winning_shares(totals-expected_pd)*size/audit_sims
    max_error = float(np.max(np.abs(audit_chances-1/count)))
    lock = {
        'version':'fair-moneyball-event-v2','config':config,'config_sha256':digest(config),'model_sha256':digest(model),'model':model,
        'seed':seed,'margin_probabilities':p.tolist(),'player_total_margin_pmf':pmfs.tolist(),
        'player_percentiles':pct.tolist(),'balance_logit_offsets':offsets.tolist(),'expected_point_differentials':expected_pd.tolist(),
        'entry_fee_mxn':100,'prizes_mxn':[100*count],
        'tie_policy':'Exact score ties share the total prizes for the positions occupied.',
        'fairness':{
            'target_first_place_chance_percent':100/count,'calibration_simulations':calibration_sims,'audit_simulations':audit_sims,
            'unbalanced_percentile_chances':public_chances(config['players'],base_chances,calibration_sims),
            'calibration_chances':public_chances(config['players'],train_chances,calibration_sims),
            'independent_audit_chances':public_chances(config['players'],audit_chances,audit_sims),
            'raw_points_above_expectation_chances':public_chances(config['players'],points_chances,audit_sims),
            'max_audit_deviation_percentage_points':max_error*100,
            'simulation_fairness_check_passed':bool(max_error <= .005),
            'adjustment_iterations':len(history),'optimization_history':history,
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
    count = len(ids)
    totals = np.zeros(count,dtype=int)
    wins = np.zeros(count,dtype=int)
    goal = lock['model']['goal_points']
    for m in event['matches']:
        s1,s2 = m.get('score1'),m.get('score2')
        if isinstance(s1,bool) or isinstance(s2,bool) or not isinstance(s1,int) or not isinstance(s2,int) or min(s1,s2) < 0 or abs(s1-s2) < 2 or max(s1,s2) < goal or (max(s1,s2) > goal and abs(s1-s2) != 2):
            raise ValueError(f'All games need completed scores to {goal}, win by two; extended games must finish two points apart.')
        margin = s1-s2
        if abs(margin)>goal:
            raise ValueError(f'A finished game to {goal} cannot have a margin above {goal}.')
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
    expected_wins=np.zeros(count)
    games=np.zeros(count,dtype=int)
    for j,m in enumerate(lock['config']['matches']):
        for pid in m['team1']+m['team2']:
            games[index[pid]]+=1
        for pid in m['team1']:
            expected_wins[index[pid]]+=team1_win_probability[j]
        for pid in m['team2']:
            expected_wins[index[pid]]+=1-team1_win_probability[j]
    for i,p in enumerate(lock['config']['players']):
        u=float(percentiles[i,totals[i]+goal*games[i]])
        u=float(np.clip(u,1e-12,1-1e-12))
        z=np.log(u)-np.log1p(-u)+lock['balance_logit_offsets'][i]
        fair_score=float(100/(1+np.exp(-z)))
        rows.append({'player_id':p['id'],'player':p['name'],'wins':int(wins[i]),'losses':int(games[i]-wins[i]),
                     'expected_wins':float(expected_wins[i]),
                     'actual_point_differential':int(totals[i]),
                     'expected_point_differential':float(lock['expected_point_differentials'][i]),
                     'performance_percentile':u*100,'moneyball_score':fair_score,
                     'simulated_first_place_chance_percent':lock['fairness']['independent_audit_chances'][i]['first_place_chance_percent']})
    rows.sort(key=lambda r:r['moneyball_score'],reverse=True)
    j=0
    prizes=lock['prizes_mxn']+[0]*(count-len(lock['prizes_mxn']))
    while j<count:
        end=j+1
        while end<count and abs(rows[end]['moneyball_score']-rows[j]['moneyball_score']) < 1e-10:
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
    return {'event_name':event['name'],'method':'Fair Play Moneyball v1','entry_fee_mxn':100,'prize_pool_mxn':sum(lock['prizes_mxn']),
            'standings':rows,'fairness':lock['fairness'],'holdout_validation':lock['model']['validation'],
            'setup_lock_sha256':lock['lock_sha256'],'experimental':True,'individual_ratings_displayed':False}


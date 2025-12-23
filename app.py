import streamlit as st
import pandas as pd
from supabase import create_client, Client
import time
from datetime import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide")

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# --- CONFIGURATION ---
K_FACTOR = 32
CLUB_ID = "tres_palapas" 

# --- DATABASE CONNECTION ---
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

# --- DATA LOADER ---
def load_data():
    # Fetch fresh data every time (No cache for live inputs)
    p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
    df_players = pd.DataFrame(p_response.data)
    
    m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=True).limit(500).execute()
    df_matches = pd.DataFrame(m_response.data)
    
    # Map Helpers
    if not df_players.empty:
        id_to_name = dict(zip(df_players['id'], df_players['name']))
        name_to_id = dict(zip(df_players['name'], df_players['id']))
    else:
        id_to_name = {}
        name_to_id = {}
        df_players = pd.DataFrame(columns=['id', 'name', 'rating', 'wins', 'losses'])

    # Map Match Display Names
    if not df_matches.empty:
        df_matches['p1'] = df_matches['t1_p1'].map(id_to_name)
        df_matches['p2'] = df_matches['t1_p2'].map(id_to_name)
        df_matches['p3'] = df_matches['t2_p1'].map(id_to_name)
        df_matches['p4'] = df_matches['t2_p2'].map(id_to_name)
        
    return df_players, df_matches, name_to_id

# --- LOGIC ENGINES ---
def calculate_hybrid_elo(t1_avg, t2_avg, score_t1, score_t2):
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1
    total_points = score_t1 + score_t2
    if total_points == 0: return 0, 0 
    raw_delta_t1 = K_FACTOR * 2 * ((score_t1 / total_points) - expected_t1)
    raw_delta_t2 = K_FACTOR * 2 * ((score_t2 / total_points) - expected_t2)
    final_t1 = max(0, raw_delta_t1) if score_t1 > score_t2 else raw_delta_t1
    final_t2 = raw_delta_t2 if score_t1 > score_t2 else max(0, raw_delta_t2)
    if score_t1 > score_t2 and final_t1 == 0: final_t1 = 1.0
    if score_t2 > score_t1 and final_t2 == 0: final_t2 = 1.0
    return final_t1, final_t2

def get_match_schedule(format_type, players):
    if len(players) < int(format_type.split('-')[0]): return []
    if format_type == "12-Player":
        # Fixed schedule 
        raw = [[([2, 5], [3, 10]), ([4, 6], [8, 9]), ([11, 0], [1, 7])], [([5, 8], [6, 2]), ([7, 9], [0, 1]), ([11, 3], [4, 10])], [([10, 1], [3, 4]), ([11, 6], [7, 2]), ([8, 0], [9, 5])], [([11, 9], [10, 5]), ([0, 3], [1, 8]), ([2, 4], [6, 7])], [([3, 6], [4, 0]), ([5, 7], [9, 10]), ([11, 1], [2, 8])], [([8, 10], [1, 2]), ([11, 4], [5, 0]), ([6, 9], [7, 3])], [([11, 7], [8, 3]), ([9, 1], [10, 6]), ([0, 2], [4, 5])], [([1, 4], [2, 9]), ([3, 5], [7, 8]), ([11, 10], [0, 6])], [([6, 8], [10, 0]), ([4, 7], [5, 1]), ([11, 2], [3, 9])], [([11, 5], [6, 1]), ([9, 0], [2, 3]), ([7, 10], [8, 4])], [([10, 2], [0, 7]), ([11, 8], [9, 4]), ([1, 3], [5, 6])]]
        matches = []
        for r_idx, round_pairs in enumerate(raw):
            for m_idx, (t1_idx, t2_idx) in enumerate(round_pairs):
                matches.append({'desc': f"R{r_idx+1}", 't1': [players[t1_idx[0]], players[t1_idx[1]]], 't2': [players[t2_idx[0]], players[t2_idx[1]]] })
        return matches
    
    p = players
    if format_type == "4-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif format_type == "5-Player": return [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5'}]
    elif format_type == "6-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[4]],'desc':'R1'}, {'t1':[p[2],p[5]],'t2':[p[0],p[4]],'desc':'R2'}, {'t1':[p[1],p[3]],'t2':[p[4],p[5]],'desc':'R3'}, {'t1':[p[0],p[5]],'t2':[p[1],p[2]],'desc':'R4'}, {'t1':[p[0],p[3]],'t2':[p[1],p[4]],'desc':'R5'}]
    elif format_type == "8-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1 A'}, {'t1':[p[4],p[5]],'t2':[p[6],p[7]],'desc':'R1 B'}, {'t1':[p[0],p[2]],'t2':[p[4],p[6]],'desc':'R2 A'}, {'t1':[p[1],p[3]],'t2':[p[5],p[7]],'desc':'R2 B'}, {'t1':[p[0],p[3]],'t2':[p[5],p[6]],'desc':'R3 A'}, {'t1':[p[1],p[2]],'t2':[p[4],p[7]],'desc':'R3 B'}, {'t1':[p[0],p[4]],'t2':[p[1],p[5]],'desc':'R4 A'}, {'t1':[p[2],p[6]],'t2':[p[3],p[7]],'desc':'R4 B'}]
    return []

# --- DB HELPERS ---
def safe_add_player(name, rating):
    try:
        supabase.table("players").insert({
            "club_id": CLUB_ID, "name": name, 
            "rating": rating * 400, "starting_rating": rating * 400
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

def process_matches(match_list, name_to_id, df_p):
    db_matches = []
    # Use a dictionary to track accumulated changes for players locally
    p_updates = {} 

    for m in match_list:
        p1, p2 = name_to_id.get(m['t1_p1']), name_to_id.get(m['t1_p2'])
        p3, p4 = name_to_id.get(m['t2_p1']), name_to_id.get(m['t2_p2'])
        if not all([p1, p2, p3, p4]): continue

        # Get ratings from DF or local accumulation
        def current_r(pid):
            if pid in p_updates: return p_updates[pid]['r']
            return float(df_p[df_p['id'] == pid]['rating'].iloc[0])

        r1, r2, r3, r4 = current_r(p1), current_r(p2), current_r(p3), current_r(p4)
        s1, s2 = m['s1'], m['s2']
        dt1, dt2 = calculate_hybrid_elo((r1+r2)/2, (r3+r4)/2, s1, s2)
        
        # Save Match
        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": m['league'],
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, "elo_delta": dt1 if s1 > s2 else dt2,
            "match_type": m['type']
        })

        # Update Local Tracking
        for pid, delta, won in [(p1, dt1, s1>s2), (p2, dt1, s1>s2), (p3, dt2, s2>s1), (p4, dt2, s2>s1)]:
            if pid not in p_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                p_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            p_updates[pid]['r'] += delta
            p_updates[pid]['mp'] += 1
            if won: p_updates[pid]['w'] += 1
            else: p_updates[pid]['l'] += 1

    # Batch Insert Matches
    if db_matches: supabase.table("matches").insert(db_matches).execute()
    
    # Update Players (One by one - Supabase bulk update is tricky without custom SQL)
    for pid, stats in p_updates.items():
        supabase.table("players").update({
            "rating": stats['r'], "wins": stats['w'], 
            "losses": stats['l'], "matches_played": stats['mp']
        }).eq("id", pid).execute()

# --- MAIN APP ---
df_players, df_matches, name_to_id = load_data()

st.sidebar.title("JUPR Leagues 🌵")
if not st.session_state.admin_logged_in:
    with st.sidebar.expander("🔒 Admin Login"):
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets["supabase"]["admin_password"]:
                st.session_state.admin_logged_in = True
                st.rerun()
            else: st.error("Wrong PW")
else:
    st.sidebar.success("Logged In: Admin")
    if st.sidebar.button("Log Out"):
        st.session_state.admin_logged_in = False
        st.rerun()

nav = ["🏆 Leaderboards", "🔍 Player Search"]
if st.session_state.admin_logged_in: nav += ["──────────", "🏟️ Live Court Manager", "🔄 Pop-Up RR", "👥 Players"]
sel = st.sidebar.radio("Go to:", nav)

# --- TABS ---
if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")
    if not df_players.empty:
        df_disp = df_players.copy()
        df_disp['JUPR'] = (df_disp['rating']/400).map('{:,.3f}'.format)
        df_disp['Win %'] = (df_disp['wins'] / df_disp['matches_played'].replace(0,1) * 100).map('{:.1f}%'.format)
        st.dataframe(df_disp[['name', 'JUPR', 'matches_played', 'wins', 'losses', 'Win %']].sort_values('rating', ascending=False), use_container_width=True, hide_index=True)

elif sel == "🔍 Player Search":
    st.header("🔍 Player History")
    p = st.selectbox("Search", [""] + sorted(df_players['name']))
    if p and not df_matches.empty:
        mask = (df_matches['p1'] == p) | (df_matches['p2'] == p) | (df_matches['p3'] == p) | (df_matches['p4'] == p)
        h = df_matches[mask].copy()
        if not h.empty:
            h['Res'] = h.apply(lambda r: "✅" if (p in [r['p1'],r['p2']] and r['score_t1']>r['score_t2']) or (p in [r['p3'],r['p4']] and r['score_t2']>r['score_t1']) else "❌", axis=1)
            st.dataframe(h[['date', 'league', 'Res', 'score_t1', 'score_t2', 'p1', 'p2', 'p3', 'p4']], use_container_width=True)

elif sel == "🏟️ Live Court Manager":
    st.header("🏟️ Live Court Manager")
    
    # State Persistence for Form
    if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
    
    with st.form("setup_lc"):
        num = st.number_input("Courts", 1, 10, st.session_state.lc_courts)
        c_data = []
        for i in range(num):
            c1, c2 = st.columns([1,3])
            t = c1.selectbox(f"T{i}", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
            n = c2.text_area(f"N{i}", height=70, placeholder="Names separated by comma or new line")
            c_data.append({'type':t, 'names':n})
        
        if st.form_submit_button("Generate Schedule"):
            # 1. Check Missing
            all_n = []
            for c in c_data: all_n.extend([x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()])
            missing = [x for x in all_n if x not in name_to_id]
            
            if missing:
                st.session_state.missing = missing
                st.rerun() # Force rerun to show missing form
            else:
                st.session_state.lc_schedule = []
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

    # Interceptor
    if 'missing' in st.session_state:
        st.warning("⚠️ New Players Detected! Add them to continue.")
        with st.form("add_missing"):
            cols = st.columns(3)
            new_inputs = {}
            for i, name in enumerate(st.session_state.missing):
                new_inputs[name] = cols[i%3].number_input(f"{name}", 1.0, 7.0, 3.0)
            
            if st.form_submit_button("Save New Players"):
                errs = []
                for name, r in new_inputs.items():
                    ok, msg = safe_add_player(name, r)
                    if not ok: errs.append(f"{name}: {msg}")
                
                if errs: st.error("\n".join(errs))
                else: 
                    st.success("Saved! Click Generate Schedule again.")
                    del st.session_state.missing
                    time.sleep(1)
                    st.rerun()

    if 'lc_schedule' in st.session_state:
        st.divider()
        with st.form("scores_lc"):
            all_res = []
            for c in st.session_state.lc_schedule:
                st.markdown(f"**Court {c['c']}**")
                for i, m in enumerate(c['m']):
                    c1, c2, c3, c4 = st.columns([3,1,1,3])
                    c1.text(f"{m['t1'][0]} & {m['t1'][1]}")
                    s1 = c2.number_input("S1", 0, key=f"lc_{c['c']}_{i}_1")
                    s2 = c3.number_input("S2", 0, key=f"lc_{c['c']}_{i}_2")
                    c4.text(f"{m['t2'][0]} & {m['t2'][1]}")
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(datetime.now()), 'league':'Live Court', 'type':f"C{c['c']} RR"})
            
            if st.form_submit_button("Submit Scores"):
                valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                if valid:
                    process_matches(valid, name_to_id, df_players)
                    st.success(f"Processed {len(valid)} matches!")
                    del st.session_state.lc_schedule
                    time.sleep(1)
                    st.rerun()

elif sel == "🔄 Pop-Up RR":
    st.header("🔄 Pop-Up Round Robin")
    
    # PERSISTENT FORM STATE
    if 'rr_courts' not in st.session_state: st.session_state.rr_courts = 1
    
    with st.form("setup_rr"):
        st.session_state.rr_courts = st.number_input("Courts", 1, 10, st.session_state.rr_courts)
        date_rr = st.date_input("Date", datetime.now())
        c_data = []
        for i in range(st.session_state.rr_courts):
            c1, c2 = st.columns([1,3])
            t = c1.selectbox(f"Format {i}", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
            n = c2.text_area(f"Players {i}", height=70, placeholder="Names...")
            c_data.append({'type':t, 'names':n})
        
        if st.form_submit_button("Generate Schedule"):
            # Check Missing
            all_n = []
            for c in c_data: all_n.extend([x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()])
            missing = [x for x in all_n if x not in name_to_id]
            
            if missing:
                st.session_state.missing_rr = missing
                st.rerun()
            else:
                st.session_state.rr_schedule = []
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.rr_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

    # Interceptor for RR
    if 'missing_rr' in st.session_state:
        st.warning("⚠️ New Players Detected! Add them to continue.")
        with st.form("add_missing_rr"):
            cols = st.columns(3)
            new_inputs = {}
            for i, name in enumerate(st.session_state.missing_rr):
                new_inputs[name] = cols[i%3].number_input(f"{name}", 1.0, 7.0, 3.0, key=f"rr_{name}")
            
            if st.form_submit_button("Save New Players"):
                for name, r in new_inputs.items():
                    safe_add_player(name, r)
                st.success("Saved! Click Generate Schedule again.")
                del st.session_state.missing_rr
                time.sleep(1)
                st.rerun()

    if 'rr_schedule' in st.session_state:
        st.divider()
        with st.form("scores_rr"):
            all_res = []
            for c in st.session_state.rr_schedule:
                st.markdown(f"**Court {c['c']}**")
                for i, m in enumerate(c['m']):
                    c1, c2, c3, c4 = st.columns([3,1,1,3])
                    c1.text(f"{m['t1'][0]} & {m['t1'][1]}")
                    s1 = c2.number_input("S1", 0, key=f"rr_{c['c']}_{i}_1")
                    s2 = c3.number_input("S2", 0, key=f"rr_{c['c']}_{i}_2")
                    c4.text(f"{m['t2'][0]} & {m['t2'][1]}")
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(date_rr), 'league':'PopUp', 'type':f"C{c['c']} RR"})
            
            if st.form_submit_button("Submit"):
                valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                if valid:
                    process_matches(valid, name_to_id, df_players)
                    st.success(f"Processed {len(valid)} matches!")
                    del st.session_state.rr_schedule
                    time.sleep(1)
                    st.rerun()

elif sel == "👥 Players":
    st.header("Player Management")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.0)
            if st.form_submit_button("Add"):
                ok, msg = safe_add_player(n, r)
                if ok: st.success("Added!"); st.rerun()
                else: st.error(msg)
    with c2:
        to_del = st.selectbox("Delete", sorted(df_players['name']))
        if st.button("Confirm Delete"):
            supabase.table("players").delete().eq("name", to_del).eq("club_id", CLUB_ID).execute()
            st.success("Deleted"); st.rerun()
    st.dataframe(df_players, use_container_width=True)

import streamlit as st
import pandas as pd
from supabase import create_client, Client
import math
import time
from datetime import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide")

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# --- CONFIGURATION ---
K_FACTOR = 32
CLUB_ID = "tres_palapas" # The ID we set in the SQL

# --- DATABASE CONNECTION ---
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = init_supabase()

# --- DATA LOADER ---
# Fetches data from Supabase instantly
def load_data():
    # 1. Fetch Players
    p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
    df_players = pd.DataFrame(p_response.data)
    
    # 2. Fetch Matches
    m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=True).execute()
    df_matches = pd.DataFrame(m_response.data)
    
    # 3. Create Helpers
    # We need to map IDs to Names for the UI, and Names to IDs for the Database
    if not df_players.empty:
        id_to_name = dict(zip(df_players['id'], df_players['name']))
        name_to_id = dict(zip(df_players['name'], df_players['id']))
    else:
        id_to_name = {}
        name_to_id = {}
        # Init empty DF if fresh
        df_players = pd.DataFrame(columns=['id', 'name', 'rating', 'wins', 'losses'])

    # 4. Map Match IDs to Names for Display
    if not df_matches.empty:
        # Create friendly columns for display
        df_matches['p1_name'] = df_matches['t1_p1'].map(id_to_name)
        df_matches['p2_name'] = df_matches['t1_p2'].map(id_to_name)
        df_matches['p3_name'] = df_matches['t2_p1'].map(id_to_name)
        df_matches['p4_name'] = df_matches['t2_p2'].map(id_to_name)
        
    return df_players, df_matches, name_to_id

# --- MATH ENGINES ---
def calculate_hybrid_elo(t1_avg, t2_avg, score_t1, score_t2):
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1
    
    total_points = score_t1 + score_t2
    if total_points == 0: return 0, 0 
    
    raw_delta_t1 = K_FACTOR * 2 * ((score_t1 / total_points) - expected_t1)
    raw_delta_t2 = K_FACTOR * 2 * ((score_t2 / total_points) - expected_t2)
    
    final_delta_t1 = 0
    final_delta_t2 = 0
    
    if score_t1 > score_t2: 
        final_delta_t1 = max(0, raw_delta_t1)
        final_delta_t2 = raw_delta_t2
        if final_delta_t1 == 0: final_delta_t1 = 1.0
    elif score_t2 > score_t1: 
        final_delta_t1 = raw_delta_t1
        final_delta_t2 = max(0, raw_delta_t2)
        if final_delta_t2 == 0: final_delta_t2 = 1.0
        
    return final_delta_t1, final_delta_t2

# --- DB WRITERS ---
def add_new_player(name, start_rating):
    """Inserts a new player into Supabase"""
    data = {
        "club_id": CLUB_ID,
        "name": name,
        "rating": start_rating * 400,
        "starting_rating": start_rating * 400,
        "matches_played": 0, "wins": 0, "losses": 0
    }
    supabase.table("players").insert(data).execute()

def process_match_batch(match_list, name_to_id, df_players):
    """
    1. Calculates Elo changes in Python.
    2. Inserts matches to DB.
    3. Updates player ratings in DB.
    """
    
    # Prepare data for insertion
    db_matches = []
    player_updates = {} # Map ID -> {rating, wins, losses}

    for m in match_list:
        # Get IDs
        p1, p2 = name_to_id.get(m['t1_p1']), name_to_id.get(m['t1_p2'])
        p3, p4 = name_to_id.get(m['t2_p1']), name_to_id.get(m['t2_p2'])
        
        if not all([p1, p2, p3, p4]): continue # Skip if bad data

        # Get Current Ratings (from DF)
        def get_r(pid):
            return float(df_players[df_players['id'] == pid]['rating'].iloc[0])
        
        r1, r2, r3, r4 = get_r(p1), get_r(p2), get_r(p3), get_r(p4)
        
        # Calculate Delta
        s1, s2 = m['score_t1'], m['score_t2']
        dt1, dt2 = calculate_hybrid_elo((r1+r2)/2, (r3+r4)/2, s1, s2)
        
        # Add to Batch
        db_matches.append({
            "club_id": CLUB_ID,
            "date": m['date'],
            "league": m['league'],
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2,
            "elo_delta": dt1 if s1 > s2 else dt2,
            "match_type": m['match_type']
        })

        # Track Player Updates (Simple accumulator)
        # Note: In a real concurrent app, we'd use SQL triggers, but this is fine for now.
        for pid, delta, is_win in [(p1, dt1, s1>s2), (p2, dt1, s1>s2), (p3, dt2, s2>s1), (p4, dt2, s2>s1)]:
            if pid not in player_updates:
                # Init with current DB values
                row = df_players[df_players['id'] == pid].iloc[0]
                player_updates[pid] = {'rating': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            
            player_updates[pid]['rating'] += delta
            player_updates[pid]['mp'] += 1
            if is_win: player_updates[pid]['w'] += 1
            else: player_updates[pid]['l'] += 1

    # 1. Insert Matches
    if db_matches:
        supabase.table("matches").insert(db_matches).execute()

    # 2. Update Players
    for pid, stats in player_updates.items():
        supabase.table("players").update({
            "rating": stats['rating'],
            "wins": stats['w'],
            "losses": stats['l'],
            "matches_played": stats['mp']
        }).eq("id", pid).execute()

def get_match_schedule(format_type, players):
    # (Same schedule logic as before)
    if len(players) < int(format_type.split('-')[0]): return []
    if format_type == "12-Player":
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
    elif format_type == "8-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[4],p[5]],'t2':[p[6],p[7]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[4],p[6]],'desc':'R2'}, {'t1':[p[1],p[3]],'t2':[p[5],p[7]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[5],p[6]],'desc':'R3'}, {'t1':[p[1],p[2]],'t2':[p[4],p[7]],'desc':'R3'}, {'t1':[p[0],p[4]],'t2':[p[1],p[5]],'desc':'R4'}, {'t1':[p[2],p[6]],'t2':[p[3],p[7]],'desc':'R4'}]
    return []

# --- LOAD DATA (Runs once per refresh) ---
df_players, df_matches, name_to_id = load_data()

# --- SIDEBAR ---
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

nav_options = ["🏆 Leaderboards", "🔍 Player Search"]
if st.session_state.admin_logged_in:
    nav_options += ["──────────", "🏟️ Live Court Manager", "🔄 Pop-Up RR", "👥 Players"]
selection = st.sidebar.radio("Go to:", nav_options, key="main_nav")

# --- UI LOGIC ---

if selection == "🏆 Leaderboards":
    st.header("🏆 Global Leaderboards")
    if not df_players.empty:
        df_disp = df_players.copy()
        df_disp['JUPR'] = (df_disp['rating'] / 400).map('{:,.3f}'.format)
        df_disp['Win %'] = (df_disp['wins'] / (df_disp['matches_played'].replace(0, 1)) * 100).map('{:.1f}%'.format)
        
        # FIX: Sort by rating FIRST, then select the columns to display
        df_sorted = df_disp.sort_values(by='rating', ascending=False)
        
        st.dataframe(
            df_sorted[['name', 'JUPR', 'matches_played', 'wins', 'losses', 'Win %']], 
            use_container_width=True, hide_index=True
        )

elif selection == "🔍 Player Search":
    st.header("🔍 Player History")
    p_name = st.selectbox("Search Player", [""] + sorted(df_players['name'].tolist()))
    if p_name and not df_matches.empty:
        # Filter matches where this player name appears
        mask = (df_matches['p1_name'] == p_name) | (df_matches['p2_name'] == p_name) | \
               (df_matches['p3_name'] == p_name) | (df_matches['p4_name'] == p_name)
        p_history = df_matches[mask].copy()
        
        if not p_history.empty:
            p_history['Result'] = p_history.apply(lambda r: "✅ Win" if (p_name in [r['p1_name'], r['p2_name']] and r['score_t1'] > r['score_t2']) or (p_name in [r['p3_name'], r['p4_name']] and r['score_t2'] > r['score_t1']) else "❌ Loss", axis=1)
            p_history['Δ JUPR'] = (p_history['elo_delta'] / 400).map('{:+.3f}'.format)
            
            st.dataframe(p_history[['date', 'league', 'Result', 'score_t1', 'score_t2', 'Δ JUPR', 'p1_name', 'p2_name', 'p3_name', 'p4_name']], use_container_width=True, hide_index=True)
        else:
            st.info("No matches found.")

elif selection == "🏟️ Live Court Manager":
    st.header("🏟️ Live Court Manager")
    
    # 1. Setup
    with st.form("setup"):
        num = st.number_input("Courts", 1, 20, 1)
        league = st.text_input("League Name", "Fall Ladder")
        date = st.date_input("Date", datetime.now())
        
        court_inputs = []
        for i in range(num):
            c1, c2 = st.columns([1,3])
            with c1: t = st.selectbox(f"Type {i+1}", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
            with c2: n = st.text_area(f"Names {i+1}", height=70)
            court_inputs.append({'type':t, 'names':n})
        
        if st.form_submit_button("Generate"):
            # Check for missing players
            all_names = []
            for c in court_inputs:
                all_names.extend([x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()])
            
            missing = [n for n in all_names if n not in name_to_id]
            if missing:
                st.session_state.missing_players = missing
            else:
                st.session_state.schedule = []
                for idx, c in enumerate(court_inputs):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.schedule.append({'court':idx+1, 'matches':get_match_schedule(c['type'], pl)})
                st.rerun()

    # 2. Missing Player Interceptor
    if 'missing_players' in st.session_state and st.session_state.missing_players:
        st.warning(f"⚠️ Found new players! Please add them.")
        with st.form("add_missing"):
            for mp in st.session_state.missing_players:
                st.number_input(f"Rating for {mp}", 1.0, 7.0, 3.0, key=f"new_{mp}")
            
            if st.form_submit_button("Save New Players"):
                for mp in st.session_state.missing_players:
                    val = st.session_state[f"new_{mp}"]
                    add_new_player(mp, val)
                del st.session_state.missing_players
                st.success("Saved! Click Generate again.")
                st.rerun()

    # 3. Score Entry
    if st.session_state.get('schedule'):
        st.divider()
        with st.form("scores"):
            for c in st.session_state.schedule:
                st.markdown(f"**Court {c['court']}**")
                for i, m in enumerate(c['matches']):
                    c1, c2, c3, c4 = st.columns([3,1,1,3])
                    c1.text(f"{m['t1'][0]} & {m['t1'][1]}")
                    s1 = c2.number_input("S1", 0, key=f"s_{c['court']}_{i}_1")
                    s2 = c3.number_input("S2", 0, key=f"s_{c['court']}_{i}_2")
                    c4.text(f"{m['t2'][0]} & {m['t2'][1]}")
            
            if st.form_submit_button("Submit"):
                matches_to_save = []
                for c in st.session_state.schedule:
                    for i, m in enumerate(c['matches']):
                        s1 = st.session_state.get(f"s_{c['court']}_{i}_1", 0)
                        s2 = st.session_state.get(f"s_{c['court']}_{i}_2", 0)
                        if s1==0 and s2==0: continue
                        matches_to_save.append({
                            't1_p1': m['t1'][0], 't1_p2': m['t1'][1],
                            't2_p1': m['t2'][0], 't2_p2': m['t2'][1],
                            'score_t1': s1, 'score_t2': s2,
                            'date': str(date), 'league': league, 'match_type': 'Live Court'
                        })
                
                process_match_batch(matches_to_save, name_to_id, df_players)
                st.success("✅ Saved to Database!")
                time.sleep(1)
                st.rerun()

elif selection == "👥 Players":
    st.header("Player Management")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("new_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.0)
            if st.form_submit_button("Add"):
                add_new_player(n, r)
                st.success("Added!")
                st.rerun()
    with c2:
        to_del = st.selectbox("Delete Player", sorted(df_players['name']))
        if st.button("Delete"):
            supabase.table("players").delete().eq("name", to_del).eq("club_id", CLUB_ID).execute()
            st.success("Deleted.")
            st.rerun()
    
    st.dataframe(df_players, use_container_width=True)

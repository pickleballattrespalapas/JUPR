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

# --- DATABASE CONNECTION (STABLE) ---
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    # Reverting to standard connection which is most stable
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

# --- LOGIC ENGINES ---
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

# --- DATA LOADER (WITH RETRY) ---
def load_data():
    # We use a retry loop to handle random network timeouts gracefully
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. Players (Overall)
            p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
            df_players = pd.DataFrame(p_response.data)
            
            # 2. League Ratings (Islands)
            l_response = supabase.table("league_ratings").select("*").eq("club_id", CLUB_ID).execute()
            df_leagues = pd.DataFrame(l_response.data)

            # 3. Matches
            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=True).limit(1000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
            # Map Helpers
            if not df_players.empty:
                id_to_name = dict(zip(df_players['id'], df_players['name']))
                name_to_id = dict(zip(df_players['name'], df_players['id']))
            else:
                id_to_name = {}
                name_to_id = {}
                df_players = pd.DataFrame(columns=['id', 'name', 'rating', 'wins', 'losses'])

            if not df_matches.empty:
                df_matches['p1'] = df_matches['t1_p1'].map(id_to_name)
                df_matches['p2'] = df_matches['t1_p2'].map(id_to_name)
                df_matches['p3'] = df_matches['t2_p1'].map(id_to_name)
                df_matches['p4'] = df_matches['t2_p2'].map(id_to_name)
                
            return df_players, df_leagues, df_matches, name_to_id, id_to_name
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1) # Wait 1s and retry
                continue
            else:
                st.error(f"⚠️ Network unstable. Please refresh. ({e})")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

# --- HELPERS ---
def get_match_schedule(format_type, players):
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
    elif format_type == "8-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1 A'}, {'t1':[p[4],p[5]],'t2':[p[6],p[7]],'desc':'R1 B'}, {'t1':[p[0],p[2]],'t2':[p[4],p[6]],'desc':'R2 A'}, {'t1':[p[1],p[3]],'t2':[p[5],p[7]],'desc':'R2 B'}, {'t1':[p[0],p[3]],'t2':[p[5],p[6]],'desc':'R3 A'}, {'t1':[p[1],p[2]],'t2':[p[4],p[7]],'desc':'R3 B'}, {'t1':[p[0],p[4]],'t2':[p[1],p[5]],'desc':'R4 A'}, {'t1':[p[2],p[6]],'t2':[p[3],p[7]],'desc':'R4 B'}]
    return []

def safe_add_player(name, rating):
    try:
        supabase.table("players").insert({
            "club_id": CLUB_ID, "name": name, 
            "rating": rating * 400, "starting_rating": rating * 400
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

# --- DUAL-TRACK PROCESSOR (OVERALL + ISLAND) ---
def process_matches(match_list, name_to_id, df_p, df_l):
    db_matches = []
    
    # Trackers for Bulk Updates
    overall_updates = {} # {pid: {r, w, l, mp}}
    island_updates = {}  # { (pid, league): {r, w, l, mp} }

    # Helper to get current Island Rating
    def get_island_r(pid, league):
        if (pid, league) in island_updates: return island_updates[(pid, league)]['r']
        # Check loaded DF
        if not df_l.empty:
            match = df_l[(df_l['player_id'] == pid) & (df_l['league_name'] == league)]
            if not match.empty: return float(match.iloc[0]['rating'])
        # Fallback to Overall Rating (or starting) if no island history
        overall = df_p[df_p['id'] == pid]
        if not overall.empty: return float(overall.iloc[0]['rating'])
        return 1200.0

    # Helper to get current Overall Rating
    def get_overall_r(pid):
        if pid in overall_updates: return overall_updates[pid]['r']
        row = df_p[df_p['id'] == pid]
        if not row.empty: return float(row.iloc[0]['rating'])
        return 1200.0

    for m in match_list:
        p1, p2, p3, p4 = [name_to_id.get(m[k]) for k in ['t1_p1', 't1_p2', 't2_p1', 't2_p2']]
        if not all([p1, p2, p3, p4]): continue
        
        s1, s2 = m['s1'], m['s2']
        league = m['league']
        
        # --- 1. OVERALL MATH ---
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2)
        
        # --- 2. ISLAND MATH ---
        ri1, ri2, ri3, ri4 = get_island_r(p1, league), get_island_r(p2, league), get_island_r(p3, league), get_island_r(p4, league)
        di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2)

        # Record Match (We store Overall Delta in main table, Island logic is separate)
        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": league,
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, "elo_delta": do1 if s1 > s2 else do2,
            "match_type": m['type']
        })

        # ACCUMULATE UPDATES
        win = s1 > s2
        participants = [(p1, do1, di1, win), (p2, do1, di1, win), (p3, do2, di2, not win), (p4, do2, di2, not win)]
        
        for pid, d_overall, d_island, won in participants:
            # Overall Update
            if pid not in overall_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                overall_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            overall_updates[pid]['r'] += d_overall
            overall_updates[pid]['mp'] += 1
            if won: overall_updates[pid]['w'] += 1
            else: overall_updates[pid]['l'] += 1
            
            # Island Update
            key = (pid, league)
            if key not in island_updates:
                # Need initial values
                curr = get_island_r(pid, league) # This fetches DB or Overall fallback
                island_updates[key] = {'r': curr, 'w': 0, 'l': 0, 'mp': 0}
            
            island_updates[key]['r'] += d_island
            island_updates[key]['mp'] += 1
            if won: island_updates[key]['w'] += 1
            else: island_updates[key]['l'] += 1

    # EXECUTE WRITES
    if db_matches: supabase.table("matches").insert(db_matches).execute()
    
    # Write Overall
    for pid, stats in overall_updates.items():
        supabase.table("players").update({
            "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']
        }).eq("id", pid).execute()

    # Write Islands (Upsert)
    island_data = []
    for (pid, league), stats in island_updates.items():
        island_data.append({
            "player_id": pid, "club_id": CLUB_ID, "league_name": league,
            "rating": stats['r'], "matches_played": stats['mp'], "wins": stats['w'], "losses": stats['l']
        })
    
    for row in island_data:
        # Check if exists to get current counts
        existing = supabase.table("league_ratings").select("*").eq("player_id", row['player_id']).eq("league_name", row['league_name']).execute()
        if existing.data:
            cur = existing.data[0]
            row['wins'] += cur['wins']
            row['losses'] += cur['losses']
            row['matches_played'] += cur['matches_played']
            supabase.table("league_ratings").update(row).eq("id", cur['id']).execute()
        else:
            supabase.table("league_ratings").insert(row).execute()

# --- MAIN APP ---
df_players, df_leagues, df_matches, name_to_id, id_to_name = load_data()

st.sidebar.title("JUPR Leagues 🌵")
if not st.session_state.admin_logged_in:
    with st.sidebar.expander("🔒 Admin Login"):
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets["supabase"]["admin_password"]:
                st.session_state.admin_logged_in = True
                st.rerun()
else:
    st.sidebar.success("Logged In: Admin")
    if st.sidebar.button("Log Out"):
        st.session_state.admin_logged_in = False
        st.rerun()

nav = ["🏆 Leaderboards", "🔍 Player Search"]
if st.session_state.admin_logged_in: 
    nav += ["──────────", "🏟️ Live Court Manager", "🔄 Pop-Up RR", "👥 Players", "📝 Match Log", "⚙️ Admin Tools"]
sel = st.sidebar.radio("Go to:", nav, key="main_nav")

# --- UI LOGIC ---

if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")
    
    # 1. League Selector
    available_leagues = ["OVERALL"]
    if not df_leagues.empty:
        unique_l = sorted(df_leagues['league_name'].unique().tolist())
        available_leagues += unique_l
        
    target_league = st.selectbox("Select View", available_leagues)
    
    # 2. Filter Data
    if target_league == "OVERALL":
        display_df = df_players.copy()
    else:
        # Filter for Island Data
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            # Merge name back in
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
    # 3. Render
    if not display_df.empty and 'rating' in display_df.columns:
        display_df['JUPR'] = (display_df['rating']/400).map('{:,.3f}'.format)
        display_df['Win %'] = (display_df['wins'] / display_df['matches_played'].replace(0,1) * 100).map('{:.1f}%'.format)
        df_sorted = display_df.sort_values('rating', ascending=False)
        st.dataframe(df_sorted[['name', 'JUPR', 'matches_played', 'wins', 'losses', 'Win %']], use_container_width=True, hide_index=True)
    else:
        st.info("No data for this league yet.")

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
    if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
    
    with st.form("setup_lc"):
        num = st.number_input("Courts", 1, 10, st.session_state.lc_courts)
        lg = st.text_input("League Name (Controls Island Rating)", "Tuesday Ladder")
        c_data = []
        for i in range(num):
            c1, c2 = st.columns([1,3])
            t = c1.selectbox(f"T{i}", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
            n = c2.text_area(f"N{i}", height=70)
            c_data.append({'type':t, 'names':n})
        
        if st.form_submit_button("Generate Schedule"):
            all_n = []
            for c in c_data: all_n.extend([x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()])
            missing = [x for x in all_n if x not in name_to_id]
            
            if missing:
                st.session_state.missing = missing
                st.rerun()
            else:
                st.session_state.lc_schedule = []
                st.session_state.active_league = lg # Save league name
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

    if 'missing' in st.session_state:
        st.warning("⚠️ New Players Detected!")
        with st.form("add_missing"):
            cols = st.columns(3)
            new_inputs = {}
            for i, name in enumerate(st.session_state.missing):
                new_inputs[name] = cols[i%3].number_input(f"{name}", 1.0, 7.0, 3.0)
            if st.form_submit_button("Save"):
                for name, r in new_inputs.items(): safe_add_player(name, r)
                st.success("Saved! Click Generate again.")
                del st.session_state.missing
                time.sleep(1)
                st.rerun()

    if 'lc_schedule' in st.session_state:
        st.divider()
        st.info(f"Posting results to: **{st.session_state.get('active_league', 'Unknown')}**")
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
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(datetime.now()), 'league':st.session_state.get('active_league', 'Unknown'), 'type':f"C{c['c']} RR"})
            
            if st.form_submit_button("Submit Scores"):
                valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                if valid:
                    process_matches(valid, name_to_id, df_players, df_leagues)
                    st.success("✅ Processed!")
                    del st.session_state.lc_schedule
                    time.sleep(1)
                    st.rerun()

elif sel == "🔄 Pop-Up RR":
    st.header("🔄 Pop-Up Round Robin")
    if 'rr_courts' not in st.session_state: st.session_state.rr_courts = 1
    
    with st.form("setup_rr"):
        st.session_state.rr_courts = st.number_input("Courts", 1, 10, st.session_state.rr_courts)
        date_rr = st.date_input("Date", datetime.now())
        lg_rr = st.text_input("Event Name", "PopUp Event")
        c_data = []
        for i in range(st.session_state.rr_courts):
            c1, c2 = st.columns([1,3])
            t = c1.selectbox(f"Format {i}", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
            n = c2.text_area(f"Players {i}", height=70)
            c_data.append({'type':t, 'names':n})
        
        if st.form_submit_button("Generate Schedule"):
            all_n = []
            for c in c_data: all_n.extend([x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()])
            missing = [x for x in all_n if x not in name_to_id]
            if missing:
                st.session_state.missing_rr = missing
                st.rerun()
            else:
                st.session_state.rr_schedule = []
                st.session_state.rr_league = lg_rr
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.rr_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

    if 'missing_rr' in st.session_state:
        st.warning("⚠️ New Players Detected!")
        with st.form("add_missing_rr"):
            cols = st.columns(3)
            new_inputs = {}
            for i, name in enumerate(st.session_state.missing_rr):
                new_inputs[name] = cols[i%3].number_input(f"{name}", 1.0, 7.0, 3.0, key=f"rr_{name}")
            if st.form_submit_button("Save"):
                for name, r in new_inputs.items(): safe_add_player(name, r)
                st.success("Saved! Click Generate again.")
                del st.session_state.missing_rr
                time.sleep(1)
                st.rerun()

    if 'rr_schedule' in st.session_state:
        st.divider()
        st.info(f"Posting results to: **{st.session_state.get('rr_league', 'PopUp')}**")
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
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(date_rr), 'league':st.session_state.get('rr_league', 'PopUp'), 'type':f"C{c['c']} RR"})
            if st.form_submit_button("Submit"):
                valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                if valid:
                    process_matches(valid, name_to_id, df_players, df_leagues)
                    st.success("✅ Processed!")
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

elif sel == "📝 Match Log":
    st.header("📝 Match Log")
    st.subheader("Recent Matches")
    edit_df = df_matches.head(50)[['id', 'date', 'league', 'p1', 'p2', 'p3', 'p4', 'score_t1', 'score_t2']].copy()
    st.dataframe(edit_df, use_container_width=True)
    
    st.divider()
    st.subheader("🗑️ Delete Match")
    m_id = st.number_input("Match ID", min_value=0, step=1)
    if st.button("Delete Match"):
        supabase.table("matches").delete().eq("id", m_id).execute()
        st.success("Deleted! Note: This does not auto-revert Elo points yet. Replay History feature coming.")
        time.sleep(1)
        st.rerun()

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    
    # --- TAB 1: CSV UPLOAD ---
    st.subheader("📤 Bulk Match Upload (CSV)")
    up_file = st.file_uploader("Upload CSV", type=['csv'])
    if up_file:
        df_csv = pd.read_csv(up_file)
        st.write("Preview:", df_csv.head())
        c1, c2, c3, c4 = st.columns(4)
        winner_col = c1.selectbox("Winner Column", df_csv.columns, index=0)
        loser_col = c2.selectbox("Loser Column", df_csv.columns, index=1)
        ladder_col = c3.text_input("Ladder Name", "CSV Upload")
        if st.button("Process Upload"):
            valid_batch = []
            for _, row in df_csv.iterrows():
                w, l = str(row[winner_col]).strip(), str(row[loser_col]).strip()
                if w in name_to_id and l in name_to_id:
                    valid_batch.append({'t1_p1': w, 't1_p2': None, 't2_p1': l, 't2_p2': None, 's1': 11, 's2': 0, 'date': str(datetime.now()), 'league': ladder_col, 'type': 'Ladder 1v1'})
            if valid_batch:
                process_matches(valid_batch, name_to_id, df_players, df_leagues)
                st.success(f"Uploaded {len(valid_batch)} matches!")

    st.divider()

    # --- TAB 2: RECALCULATE LEAGUE ---
    st.subheader("🔄 Recalculate League History")
    st.markdown("""
    **Use this if:**
    * You deleted a match and need to fix the points.
    * You changed a player's starting rating.
    * The math looks "off."
    """)

    # 1. Select Scope
    league_options = ["ALL (Full System Reset)"] + sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else ["ALL (Full System Reset)"]
    target_reset = st.selectbox("Select League to Recalculate", league_options)

    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching numbers... this may take a moment."):
            # A. FETCH ALL DATA needed for replay
            all_players = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute().data
            all_matches = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=False).execute().data # Oldest first
            
            # B. RESET MEMORY
            p_map = {p['id']: {'r': float(p['starting_rating']), 'w': 0, 'l': 0, 'mp': 0, 'name': p['name']} for p in all_players}
            island_map = {} # (pid, league) -> {r, w, l, mp}

            # C. REPLAY MATCHES
            for m in all_matches:
                if target_reset != "ALL (Full System Reset)" and m['league'] != target_reset:
                    continue

                p1, p2, p3, p4 = m['t1_p1'], m['t1_p2'], m['t2_p1'], m['t2_p2']
                s1, s2 = m['score_t1'], m['score_t2']
                league = m['league']
                
                # 1. Overall Math (Only if resetting ALL)
                if target_reset == "ALL (Full System Reset)":
                    ro1, ro2, ro3, ro4 = p_map[p1]['r'], p_map[p2]['r'], p_map[p3]['r'], p_map[p4]['r']
                    do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2)
                    
                    win = s1 > s2
                    for pid, delta, is_win in [(p1, do1, win), (p2, do1, win), (p3, do2, not win), (p4, do2, not win)]:
                        p_map[pid]['r'] += delta
                        p_map[pid]['mp'] += 1
                        if is_win: p_map[pid]['w'] += 1
                        else: p_map[pid]['l'] += 1

                # 2. Island Math
                def get_i_r(pid, lg):
                    if (pid, lg) not in island_map:
                        island_map[(pid, lg)] = {'r': 1200.0, 'w':0, 'l':0, 'mp':0}
                    return island_map[(pid, lg)]['r']

                ri1, ri2, ri3, ri4 = get_i_r(p1, league), get_i_r(p2, league), get_i_r(p3, league), get_i_r(p4, league)
                di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2)

                win = s1 > s2
                for pid, delta, is_win in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                    k = (pid, league)
                    if k not in island_map: island_map[k] = {'r': 1200.0, 'w':0, 'l':0, 'mp':0}
                    island_map[k]['r'] += delta
                    island_map[k]['mp'] += 1
                    if is_win: island_map[k]['w'] += 1
                    else: island_map[k]['l'] += 1

            # D. SAVE RESULTS
            if target_reset == "ALL (Full System Reset)":
                for pid, stats in p_map.items():
                    supabase.table("players").update({
                        "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']
                    }).eq("id", pid).execute()
            
            if target_reset != "ALL (Full System Reset)":
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", target_reset).execute()
            else:
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()
            
            new_islands = []
            for (pid, lg), stats in island_map.items():
                if target_reset == "ALL (Full System Reset)" or lg == target_reset:
                    new_islands.append({
                        "club_id": CLUB_ID, "player_id": pid, "league_name": lg,
                        "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']
                    })
            
            if new_islands:
                chunk_size = 1000
                for i in range(0, len(new_islands), chunk_size):
                    supabase.table("league_ratings").insert(new_islands[i:i+chunk_size]).execute()

            st.success(f"✅ Successfully replayed {len(all_matches)} matches!")
            time.sleep(2)
            st.rerun()

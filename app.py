import streamlit as st
import pandas as pd
from supabase import create_client, Client
import time
from datetime import datetime
import difflib 

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide")

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# --- CONFIGURATION ---
K_FACTOR = 32
CLUB_ID = "tres_palapas" 

# --- MAGIC LINK LOGIN ---
query_params = st.query_params
if "admin_key" in query_params:
    if query_params["admin_key"] == st.secrets["supabase"]["admin_password"]:
        st.session_state.admin_logged_in = True

# --- DATABASE CONNECTION ---
@st.cache_resource(ttl=1)
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
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

# --- DATA LOADER ---
def load_data():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. Players
            p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
            df_players = pd.DataFrame(p_response.data)
            
            # 2. Island Ratings
            l_response = supabase.table("league_ratings").select("*").eq("club_id", CLUB_ID).execute()
            df_leagues = pd.DataFrame(l_response.data)

            # 3. Matches
            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("id", desc=True).limit(50000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
            # 4. League Metadata
            meta_response = supabase.table("leagues_metadata").select("*").eq("club_id", CLUB_ID).execute()
            df_meta = pd.DataFrame(meta_response.data)

            # --- ID CLEANING ---
            if not df_players.empty:
                df_players['id'] = pd.to_numeric(df_players['id'], errors='coerce').fillna(-1).astype(int)
                id_to_name = dict(zip(df_players['id'], df_players['name']))
                name_to_id = dict(zip(df_players['name'], df_players['id']))
            else:
                id_to_name = {}
                name_to_id = {}
                df_players = pd.DataFrame(columns=['id', 'name', 'rating', 'wins', 'losses'])

            if not df_leagues.empty:
                df_leagues['player_id'] = pd.to_numeric(df_leagues['player_id'], errors='coerce').fillna(-1).astype(int)

            if not df_matches.empty:
                for col in ['t1_p1', 't1_p2', 't2_p1', 't2_p2']:
                    df_matches[col] = pd.to_numeric(df_matches[col], errors='coerce').fillna(-1).astype(int)

                df_matches['p1'] = df_matches['t1_p1'].map(id_to_name)
                df_matches['p2'] = df_matches['t1_p2'].map(id_to_name)
                df_matches['p3'] = df_matches['t2_p1'].map(id_to_name)
                df_matches['p4'] = df_matches['t2_p2'].map(id_to_name)
                
                # --- DATE PARSING (STRIP TIME) ---
                df_matches['date_str'] = df_matches['date'].astype(str).str[:10] 
                df_matches['date_obj'] = pd.to_datetime(df_matches['date_str'], errors='coerce')
                
            return df_players, df_leagues, df_matches, df_meta, name_to_id, id_to_name
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1) 
                continue
            else:
                st.error(f"⚠️ Network unstable. Please refresh. ({e})")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

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

# --- PROCESSOR ---
def process_matches(match_list, name_to_id, df_p, df_l):
    db_matches = []
    overall_updates = {} 
    island_updates = {}

    def get_island_r(pid, league):
        if (pid, league) in island_updates: return island_updates[(pid, league)]['r']
        if not df_l.empty:
            match = df_l[(df_l['player_id'] == pid) & (df_l['league_name'] == league)]
            if not match.empty: return float(match.iloc[0]['rating'])
        if pid in overall_updates: return overall_updates[pid]['r']
        overall = df_p[df_p['id'] == pid]
        if not overall.empty: return float(overall.iloc[0]['rating'])
        return 1200.0

    def get_overall_r(pid):
        if pid in overall_updates: return overall_updates[pid]['r']
        row = df_p[df_p['id'] == pid]
        if not row.empty: return float(row.iloc[0]['rating'])
        return 1200.0

    for m in match_list:
        p1, p2, p3, p4 = [name_to_id.get(m[k]) for k in ['t1_p1', 't1_p2', 't2_p1', 't2_p2']]
        if not p1 or not p3: continue
        
        s1, s2 = m['s1'], m['s2']
        league = m['league']
        is_popup = m.get('match_type') == 'PopUp' or m.get('is_popup', False)
        
        def safe_get_o(pid):
            if pid is None: return 1200.0
            return get_overall_r(pid)
            
        ro1, ro2, ro3, ro4 = safe_get_o(p1), safe_get_o(p2), safe_get_o(p3), safe_get_o(p4)
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2)
        
        di1, di2 = 0, 0
        if not is_popup:
            def safe_get_i(pid, lg):
                if pid is None: return 1200.0
                return get_island_r(pid, lg)
            
            ri1, ri2, ri3, ri4 = safe_get_i(p1, league), safe_get_i(p2, league), safe_get_i(p3, league), safe_get_i(p4, league)
            di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2)

        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": league,
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, "elo_delta": do1 if s1 > s2 else do2,
            "match_type": m['match_type']
        })

        win = s1 > s2
        participants = [(p1, do1, di1, win), (p2, do1, di1, win), (p3, do2, di2, not win), (p4, do2, di2, not win)]
        
        for pid, d_overall, d_island, won in participants:
            if pid is None: continue 
            
            if pid not in overall_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                overall_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            overall_updates[pid]['r'] += d_overall
            overall_updates[pid]['mp'] += 1
            if won: overall_updates[pid]['w'] += 1
            else: overall_updates[pid]['l'] += 1
            
            if not is_popup:
                key = (pid, league)
                if key not in island_updates:
                    curr = get_island_r(pid, league) 
                    island_updates[key] = {'r': curr, 'w': 0, 'l': 0, 'mp': 0}
                island_updates[key]['r'] += d_island
                island_updates[key]['mp'] += 1
                if won: island_updates[key]['w'] += 1
                else: island_updates[key]['l'] += 1

    if db_matches: supabase.table("matches").insert(db_matches).execute()
    
    for pid, stats in overall_updates.items():
        supabase.table("players").update({
            "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']
        }).eq("id", pid).execute()

    if island_updates:
        island_data = []
        for (pid, league), stats in island_updates.items():
            island_data.append({
                "player_id": pid, "club_id": CLUB_ID, "league_name": league,
                "rating": stats['r'], "matches_played": stats['mp'], "wins": stats['w'], "losses": stats['l']
            })
        
        for row in island_data:
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
df_players, df_leagues, df_matches, df_meta, name_to_id, id_to_name = load_data()

# --- PROCESS META ---
if not df_meta.empty:
    active_leagues_list = sorted(df_meta[df_meta['is_active'] == True]['league_name'].tolist())
else:
    active_leagues_list = []

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

# --- NAVIGATION ---
nav = ["🏆 Leaderboards", "🔍 Player Search", "❓ FAQs"]
if st.session_state.admin_logged_in: 
    nav += ["──────────", "📋 Roster Check", "🏟️ League Manager", "⚡ Batch Entry", "🔄 Pop-Up RR", "👥 Players", "📝 Match Log", "⚙️ Admin Tools", "📘 Admin Guide"]
sel = st.sidebar.radio("Go to:", nav, key="main_nav")

# --- UI LOGIC ---

if sel == "🏆 Leaderboards":
    st.header("🏆 League Leaderboards")
    
    available_leagues = ["OVERALL"]
    if not df_meta.empty:
        std_leagues = df_meta[df_meta['league_type'] != 'Pop-Up']['league_name'].unique().tolist()
        unique_l = sorted(std_leagues)
        available_leagues += unique_l
    elif not df_leagues.empty:
        unique_l = sorted([l for l in df_leagues['league_name'].unique().tolist() if "Pop" not in l])
        available_leagues += unique_l
        
    target_league = st.selectbox("Select League", available_leagues)
    
    # Threshold Logic
    db_threshold = 1
    if not df_meta.empty and target_league != "OVERALL":
        meta_row = df_meta[df_meta['league_name'] == target_league]
        if not meta_row.empty:
            db_threshold = int(meta_row.iloc[0]['min_weeks'])
    
    if target_league != "OVERALL":
        threshold_days = st.slider("Min Days Played (Filters Awards)", 1, 10, db_threshold)
    else:
        threshold_days = 1

    # 3. Data Prep for Stats
    if target_league == "OVERALL":
        base_df = df_players.copy()
        matches_scope = df_matches.copy() # Use .copy() to avoid setting on copy warnings
    else:
        if df_leagues.empty: base_df = pd.DataFrame()
        else:
            base_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            base_df['name'] = base_df['player_id'].map(id_to_name)
        
        matches_scope = df_matches[df_matches['league'] == target_league].copy()
        if matches_scope.empty:
            matches_scope = df_matches[df_matches['league'].astype(str).str.strip() == target_league.strip()].copy()

    # --- CRITICAL FIX: Ensure day_id exists locally ---
    # This prevents the KeyError if the data loader cache is stale
    if not matches_scope.empty and 'day_id' not in matches_scope.columns:
        matches_scope['day_id'] = matches_scope['date'].astype(str).str[:10]

    if not base_df.empty and 'rating' in base_df.columns:
        stats_map = {}
        
        if not matches_scope.empty:
            for _, m in matches_scope.iterrows():
                # SAFETY: Handle missing day_id even after fix attempt
                day_id = m.get('day_id', str(m.get('date', ''))[:10])
                if pd.isna(day_id) or day_id == "nan" or day_id == "": continue
                
                delta = m['elo_delta']
                t1_won = m['score_t1'] > m['score_t2']
                
                # ... rest of the loop remains the same ...
                raw_pids = [m['t1_p1'], m['t1_p2'], m['t2_p1'], m['t2_p2']]
                for i, raw_pid in enumerate(raw_pids):
                    if pd.isna(raw_pid) or raw_pid == -1: continue
                    pid_int = int(raw_pid)
                    
                    if pid_int not in stats_map: stats_map[pid_int] = {'days': set(), 'total_delta': 0.0, 'live_matches': 0}
                    stats_map[pid_int]['days'].add(day_id) 
                    stats_map[pid_int]['live_matches'] += 1
                    
                    is_t1 = (i <= 1)
                    if (is_t1 and t1_won) or (not is_t1 and not t1_won):
                        stats_map[pid_int]['total_delta'] += delta
                    else:
                        stats_map[pid_int]['total_delta'] -= delta

        # Merge
        base_df['active_days'] = base_df.apply(lambda x: len(stats_map.get(int(x['id'] if 'id' in x else x['player_id']), {}).get('days', [])), axis=1)
        base_df['rating_gain'] = base_df.apply(lambda x: stats_map.get(int(x['id'] if 'id' in x else x['player_id']), {}).get('total_delta', 0.0), axis=1)
        base_df['live_matches_count'] = base_df.apply(lambda x: stats_map.get(int(x['id'] if 'id' in x else x['player_id']), {}).get('live_matches', 0), axis=1)
        
        if target_league != "OVERALL":
             base_df['display_matches'] = base_df['live_matches_count']
        else:
             base_df['display_matches'] = base_df['matches_played']

        base_df['JUPR'] = base_df['rating'] / 400
        base_df['win_pct'] = (base_df['wins'] / base_df['display_matches'].replace(0, 1)) * 100
        
        # --- 4. SHOW GRIDS ---
        if target_league != "OVERALL":
            qualified_df = base_df[base_df['active_days'] >= threshold_days].copy()
            
            if not qualified_df.empty:
                st.markdown("### 🏅 Top Performers")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**👑 Highest Rating**")
                    top_rating = qualified_df.sort_values('rating', ascending=False).head(5)
                    for _, r in top_rating.iterrows(): st.markdown(f"**{r['JUPR']:.3f}** - {r['name']}")
                with col2:
                    st.markdown("**🔥 Most Improved**")
                    top_gain = qualified_df.sort_values('rating_gain', ascending=False).head(5)
                    for _, r in top_gain.iterrows(): 
                        val = r['rating_gain'] / 400
                        st.markdown(f"**{'+' if val>0 else ''}{val:.3f}** - {r['name']}")
                with col3:
                    st.markdown("**🎯 Best Win %**")
                    top_pct = qualified_df.sort_values('win_pct', ascending=False).head(5)
                    for _, r in top_pct.iterrows(): st.markdown(f"**{r['win_pct']:.1f}%** - {r['name']}")
                with col4:
                    st.markdown("**🚜 Most Wins**")
                    top_wins = qualified_df.sort_values('wins', ascending=False).head(5)
                    for _, r in top_wins.iterrows(): st.markdown(f"**{r['wins']} Wins** - {r['name']}")
                st.divider()
            else:
                st.info(f"ℹ️ Top 5 Awards hidden until players reach {threshold_days} active days.")

        # --- 5. FULL TABLE ---
        st.markdown("### 📊 Full Standings")
        final_view = base_df.sort_values('rating', ascending=False).copy()
        final_view['JUPR'] = final_view['JUPR'].map('{:,.3f}'.format)
        final_view['Win %'] = final_view['win_pct'].map('{:.1f}%'.format)
        final_view['Gain'] = (final_view['rating_gain']/400).map('{:+.3f}'.format)
        
        st.dataframe(
            final_view[['name', 'JUPR', 'matches_played', 'active_days', 'wins', 'losses', 'Win %', 'Gain']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "active_days": "Days Played"
            }
        )
        if target_league != "OVERALL": st.caption(f"👇 Share: `https://jupr-leagues.streamlit.app/?league={target_league.replace(' ', '%20')}`")

    else:
        st.info("No data available for this league.")

elif sel == "🔍 Player Search":
    st.header("🔍 Player History")
    p = st.selectbox("Search", [""] + sorted(df_players['name']))
    
    if p:
        st.subheader("📊 Rating Snapshot")
        p_row = df_players[df_players['name'] == p]
        summary_data = []
        if not p_row.empty:
            summary_data.append({
                "Context": "🌍 OVERALL",
                "Rating": f"{(float(p_row.iloc[0]['rating'])/400):.3f}",
                "Matches": p_row.iloc[0]['matches_played']
            })
        
        if not df_leagues.empty:
            pid = name_to_id.get(p)
            islands = df_leagues[df_leagues['player_id'] == pid]
            for _, row in islands.iterrows():
                summary_data.append({
                    "Context": f"🏝️ {row['league_name']}",
                    "Rating": f"{(float(row['rating'])/400):.3f}",
                    "Matches": row['matches_played']
                })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        if not df_matches.empty:
            mask = (df_matches['p1'] == p) | (df_matches['p2'] == p) | (df_matches['p3'] == p) | (df_matches['p4'] == p)
            h = df_matches[mask].copy()
            if not h.empty:
                def get_stats(r):
                    is_t1 = p in [r['p1'], r['p2']]
                    t1_won = r['score_t1'] > r['score_t2']
                    player_won = (is_t1 and t1_won) or (not is_t1 and not t1_won)
                    delta = r['elo_delta'] / 400
                    if not player_won: delta = -delta
                    return ("✅ Win" if player_won else "❌ Loss", delta, r['elo_delta'])

                stats = h.apply(get_stats, axis=1, result_type='expand')
                h['Result'] = stats[0]
                h['Δ JUPR'] = stats[1].map('{:+.3f}'.format)
                h['Raw Pts'] = stats[2]
                
                st.dataframe(h[['date', 'league', 'Result', 'Δ JUPR', 'Raw Pts', 'score_t1', 'score_t2', 'p1', 'p2', 'p3', 'p4']], use_container_width=True, hide_index=True)
            else:
                st.info("No matches found.")

elif sel == "❓ FAQs":
    st.header("❓ Frequently Asked Questions")
    with st.expander("🤔 What is JUPR?"):
        st.markdown("**JUPR (Just a Universal Pickleball Rating)** is a modern rating system designed specifically for our club.")
    with st.expander("📈 How are ratings calculated?"):
        st.markdown("We use an **Elo-based formula**. Winning 11-0 is worth more than winning 11-9.")
    with st.expander("🏝️ Why is my 'Overall' rating different from my 'League' rating?"):
        st.markdown("Your **Overall** rating includes every match. Your **League** rating only includes matches from that specific league.")

elif sel == "📋 Roster Check":
    st.header("📋 Roster Check")
    st.markdown("Paste a list of names to check ratings. Any names not found will be added to the 'New Player' table for onboarding.")
    
    lookup_scope = st.radio("Rating Scope", ["OVERALL"] + (sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else []), horizontal=True)
    raw_names = st.text_area("Paste Names (one per line or comma-separated)", height=100, placeholder="Tom Elliott\nKeith Brand\nNew Player")
    
    if st.button("Analyze List"):
        if 'roster_results' in st.session_state: del st.session_state.roster_results
        if 'df_new_players' in st.session_state: del st.session_state.df_new_players

        parsed = [x.strip() for x in raw_names.replace('\n',',').split(',') if x.strip()]
        
        found_data = []
        new_players = []
        
        for n in parsed:
            if n in name_to_id:
                pid = name_to_id[n]
                r_val = 1200.0
                if lookup_scope == "OVERALL":
                    row = df_players[df_players['id'] == pid]
                    if not row.empty: r_val = float(row.iloc[0]['rating'])
                else:
                    row = df_leagues[(df_leagues['player_id'] == pid) & (df_leagues['league_name'] == lookup_scope)]
                    if not row.empty: r_val = float(row.iloc[0]['rating'])
                    else:
                        ov_row = df_players[df_players['id'] == pid]
                        if not ov_row.empty: r_val = float(ov_row.iloc[0]['rating'])
                
                found_data.append({"Name": n, "Rating": f"{(r_val/400):.3f}", "Status": "✅ Found"})
            else:
                new_players.append(n)

        st.session_state.roster_results = {
            'found': found_data,
            'new': new_players
        }

    if 'roster_results' in st.session_state:
        results = st.session_state.roster_results
        if results['found']:
            st.success(f"Found {len(results['found'])} players.")
            st.dataframe(pd.DataFrame(results['found']), use_container_width=True)
        if results['new']:
            st.error(f"🛑 Found {len(results['new'])} new players.")
            if 'df_new_players' not in st.session_state:
                st.session_state.df_new_players = pd.DataFrame({"Name": results['new'], "Rating": [3.5] * len(results['new'])})
            edited_df = st.data_editor(st.session_state.df_new_players, column_config={"Rating": st.column_config.NumberColumn(min_value=1.0, max_value=7.0, step=0.1)}, hide_index=True, use_container_width=True)
            if st.button("💾 Save New Players"):
                for _, row in edited_df.iterrows():
                    safe_add_player(row['Name'], row['Rating'])
                st.success("✅ Created profiles!"); time.sleep(1); del st.session_state.roster_results; st.rerun()

# --- NEW: LEAGUE MANAGER TAB ---
elif sel == "🏟️ League Manager":
    st.header("🏟️ League Manager")
    
    lm_tabs = st.tabs(["📝 Live Match Entry", "⚙️ League Settings", "🆕 Create League", "🔄 Migration"])
    
    # --- TAB 1: LIVE MATCH ENTRY (RESTORED) ---
    with lm_tabs[0]:
        st.subheader("Live Court Management")
        
        active_opts = active_leagues_list if active_leagues_list else ["Default League"]
        lg_live = st.selectbox("Select League", active_opts, key="live_league_selector")
        
        if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
        st.session_state.lc_courts = st.number_input("Number of Courts", 1, 10, st.session_state.lc_courts, key="lc_court_input")
        
        with st.form("setup_lc"):
            c_data = []
            for i in range(st.session_state.lc_courts):
                c1, c2 = st.columns([1,3])
                t = c1.selectbox(f"Format (Court {i+1})", ["4-Player","5-Player","6-Player","8-Player","12-Player"], key=f"fmt_{i}")
                n = c2.text_area(f"Players (Court {i+1})", height=70, key=f"names_{i}", placeholder="Paste names here...")
                c_data.append({'type':t, 'names':n})
            
            if st.form_submit_button("Generate Schedule"):
                st.session_state.lc_schedule = []
                st.session_state.lc_active_league = lg_live
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

        if 'lc_schedule' in st.session_state:
            st.divider()
            st.info(f"Posting results to: **{st.session_state.get('lc_active_league', 'Unknown')}**")
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
                        all_res.append({
                            't1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1],
                            's1':s1, 's2':s2, 'date':str(datetime.now()), 
                            'league':st.session_state.get('lc_active_league', 'Unknown'), 
                            'type':f"C{c['c']} RR", 'match_type': 'Live Match', 'is_popup': False
                        })
                
                if st.form_submit_button("Submit Scores"):
                    valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                    if valid:
                        process_matches(valid, name_to_id, df_players, df_leagues)
                        st.success("✅ Processed!")
                        del st.session_state.lc_schedule
                        time.sleep(1)
                        st.rerun()

    # --- TAB 2: SETTINGS ---
    with lm_tabs[1]:
        if not df_meta.empty:
            st.info("Uncheck 'Active' to retire a league. It will be hidden from entry forms but remain in history.")
            edit_meta = df_meta[['id', 'league_name', 'league_type', 'min_weeks', 'is_active']].copy()
            edited_leagues = st.data_editor(
                edit_meta,
                column_config={
                    "league_name": "League Name",
                    "league_type": st.column_config.SelectboxColumn("Type", options=["Standard", "Pop-Up"]),
                    "min_weeks": st.column_config.NumberColumn("Min Weeks", min_value=1, max_value=20),
                    "is_active": st.column_config.CheckboxColumn("Active?")
                },
                disabled=["id", "league_name"],
                hide_index=True,
                use_container_width=True,
                key="league_editor"
            )
            if st.button("💾 Save Changes"):
                for index, row in edited_leagues.iterrows():
                    supabase.table("leagues_metadata").update({
                        "league_type": row['league_type'],
                        "min_weeks": row['min_weeks'],
                        "is_active": row['is_active']
                    }).eq("id", row['id']).execute()
                st.success("Settings Updated!"); time.sleep(1); st.rerun()
        else:
            st.warning("No leagues found. Go to 'Migration Tool'.")

    # --- TAB 3: CREATE ---
    with lm_tabs[2]:
        with st.form("create_league"):
            new_name = st.text_input("New League Name")
            new_type = st.selectbox("Type", ["Standard", "Pop-Up"])
            new_weeks = st.number_input("Min Weeks Required", 1, 20, 4)
            if st.form_submit_button("Create League"):
                if new_name:
                    try:
                        supabase.table("leagues_metadata").insert({
                            "club_id": CLUB_ID, "league_name": new_name, 
                            "league_type": new_type, "min_weeks": new_weeks, "is_active": True
                        }).execute()
                        st.success(f"Created {new_name}!"); time.sleep(1); st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- TAB 4: MIGRATION ---
    with lm_tabs[3]:
        st.write("### 🔄 Scan & Sync")
        if st.button("Run Migration Script"):
            unique_match_leagues = df_matches['league'].unique().tolist()
            existing_meta_leagues = df_meta['league_name'].unique().tolist() if not df_meta.empty else []
            created_count = 0
            for lg in unique_match_leagues:
                if lg not in existing_meta_leagues:
                    is_popup = "Pop" in lg
                    supabase.table("leagues_metadata").insert({
                        "club_id": CLUB_ID,
                        "league_name": lg,
                        "league_type": "Pop-Up" if is_popup else "Standard",
                        "min_weeks": 1 if is_popup else 4,
                        "is_active": True
                    }).execute()
                    created_count += 1
            if created_count > 0:
                st.success(f"✅ Migrated {created_count} new leagues!"); time.sleep(2); st.rerun()
            else:
                st.info("System is up to date.")

elif sel == "⚡ Batch Entry":
    st.header("⚡ Batch Match Entry")
    
    entry_date = st.date_input("Match Date", datetime.now(), help="Pick the date these matches actually happened.")
    
    if active_leagues_list:
        batch_league = st.selectbox("Select League", active_leagues_list)
    else:
        batch_league = st.text_input("League Name (System Empty)", "Fall 2025 Ladder")
        st.warning("⚠️ No active leagues found in metadata. Run 'Migration Tool' in League Manager.")

    player_list = sorted(df_players['name'].tolist())
    
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = pd.DataFrame(
            [{'T1_P1': None, 'T1_P2': None, 'Score_1': 0, 'Score_2': 0, 'T2_P1': None, 'T2_P2': None} for _ in range(5)]
        )

    edited_batch = st.data_editor(
        st.session_state.batch_df,
        column_config={
            "T1_P1": st.column_config.SelectboxColumn("Team 1 - P1", options=player_list, required=True),
            "T1_P2": st.column_config.SelectboxColumn("Team 1 - P2", options=player_list),
            "Score_1": st.column_config.NumberColumn("Score 1", min_value=0, max_value=30, step=1),
            "Score_2": st.column_config.NumberColumn("Score 2", min_value=0, max_value=30, step=1),
            "T2_P1": st.column_config.SelectboxColumn("Team 2 - P1", options=player_list, required=True),
            "T2_P2": st.column_config.SelectboxColumn("Team 2 - P2", options=player_list),
        },
        column_order=("T1_P1", "T1_P2", "Score_1", "Score_2", "T2_P1", "T2_P2"), 
        num_rows="dynamic",
        use_container_width=True
    )

    if st.button("Process Batch"):
        valid_batch = []
        for _, row in edited_batch.iterrows():
            if row['T1_P1'] and row['T2_P1'] and (row['Score_1'] + row['Score_2'] > 0):
                m_type = "Live Match"
                if not df_meta.empty:
                    l_row = df_meta[df_meta['league_name'] == batch_league]
                    if not l_row.empty and l_row.iloc[0]['league_type'] == "Pop-Up":
                        m_type = "PopUp"

                match_data = {
                    't1_p1': row['T1_P1'], 't1_p2': row['T1_P2'], 
                    't2_p1': row['T2_P1'], 't2_p2': row['T2_P2'], 
                    's1': int(row['Score_1']), 's2': int(row['Score_2']), 
                    'date': str(entry_date), 
                    'league': batch_league, 
                    'type': 'Batch Entry', 'match_type': m_type, 'is_popup': (m_type == "PopUp")
                }
                valid_batch.append(match_data)
        
        if valid_batch:
            process_matches(valid_batch, name_to_id, df_players, df_leagues)
            st.success(f"✅ Successfully processed {len(valid_batch)} matches for {entry_date}!")
            del st.session_state.batch_df
            time.sleep(1)
            st.rerun()

elif sel == "🔄 Pop-Up RR":
    st.header("🔄 Pop-Up Round Robin")
    
    with st.form("setup_rr"):
        date_rr = st.date_input("Date", datetime.now())
        rr_opts = active_leagues_list if active_leagues_list else ["PopUp Event"]
        lg_rr = st.selectbox("Event Name", rr_opts)
        
        c1, c2 = st.columns([1,3])
        t = c1.selectbox("Format", ["4-Player","5-Player","6-Player","8-Player","12-Player"])
        n = c2.text_area("Players", height=70)
        
        if st.form_submit_button("Generate Schedule"):
            st.session_state.rr_schedule = []
            st.session_state.rr_league = lg_rr
            pl = [x.strip() for x in n.replace('\n',',').split(',') if x.strip()]
            st.session_state.rr_schedule.append({'c': 1, 'm': get_match_schedule(t, pl)})
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
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(date_rr), 'league':st.session_state.get('rr_league', 'PopUp'), 'type':f"C{c['c']} RR", 'match_type': 'PopUp', 'is_popup': True})
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
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.0)
            if st.form_submit_button("Add Player"):
                ok, msg = safe_add_player(n, r)
                if ok: st.success(f"Added {n}!"); time.sleep(1); st.rerun()
                else: st.error(msg)
    with c2:
        st.subheader("✏️ Edit Player")
        p_edit = st.selectbox("Select Player", [""] + sorted(df_players['name']))
        if p_edit:
            curr_row = df_players[df_players['name'] == p_edit].iloc[0]
            curr_start = float(curr_row['starting_rating']) / 400
            new_name = st.text_input("Edit Name", value=p_edit)
            new_start = st.number_input("New Start Rating", 1.0, 7.0, curr_start, step=0.1)
            if st.button("Update Profile"):
                supabase.table("players").update({"name": new_name, "starting_rating": new_start * 400, "rating": new_start * 400}).eq("name", p_edit).eq("club_id", CLUB_ID).execute()
                st.success(f"Updated {new_name}!"); time.sleep(1); st.rerun()
    with c3:
        st.subheader("🗑️ Delete")
        to_del = st.selectbox("Select to Remove", [""] + sorted(df_players['name']))
        if to_del and st.button("Confirm Delete"):
            supabase.table("players").delete().eq("name", to_del).eq("club_id", CLUB_ID).execute()
            st.success("Deleted."); time.sleep(1); st.rerun()
    st.dataframe(df_players, use_container_width=True)

elif sel == "📝 Match Log":
    st.header("📝 Match Log")
    filter_type = st.radio("Filter Matches", ["All", "League Matches", "Pop-Up Events"], horizontal=True)
    if filter_type == "League Matches": view_df = df_matches[df_matches['match_type'] != 'PopUp']
    elif filter_type == "Pop-Up Events": view_df = df_matches[df_matches['match_type'] == 'PopUp']
    else: view_df = df_matches
    
    col1, col2 = st.columns([1, 4])
    id_filter = col1.number_input("Jump to ID:", min_value=0, value=0)
    if id_filter > 0: view_df = view_df[view_df['id'] == id_filter]

    st.write("### 🗑️ Bulk Delete Matches")
    st.info("Select checkboxes below to delete matches.")
    edit_df = view_df.head(5000)[['id', 'date', 'league', 'match_type', 'elo_delta', 'p1', 'p2', 'p3', 'p4', 'score_t1', 'score_t2']].copy()
    edit_df.insert(0, "Delete", False) 
    edited_log = st.data_editor(edit_df, column_config={"Delete": st.column_config.CheckboxColumn("Delete?", default=False), "elo_delta": st.column_config.NumberColumn("Elo Delta", format="%.1f")}, disabled=["id", "date", "league", "match_type", "elo_delta", "p1", "p2", "p3", "p4", "score_t1", "score_t2"], use_container_width=True, hide_index=True)
    
    to_delete = edited_log[edited_log['Delete'] == True]
    if not to_delete.empty:
        st.error(f"⚠️ You have selected {len(to_delete)} matches for deletion.")
        if st.button("Confirm Bulk Delete"):
            supabase.table("matches").delete().in_("id", to_delete['id'].tolist()).execute()
            st.success("Deleted!"); time.sleep(1); st.rerun()

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    
    # --- UPDATED: LEAGUE MERGER (FORCE) ---
    st.subheader("🔗 League Merger")
    st.markdown("Merge duplicate leagues (e.g. 'Fall 2025' and 'Fall '25').")
    
    if not df_matches.empty:
        # Get all unique names
        all_match_names = sorted(df_matches['league'].astype(str).unique().tolist())
        
        c1, c2 = st.columns(2)
        # Any existing league name in matches can be source
        from_league = c1.selectbox("Move Matches FROM:", all_match_names)
        
        # Destination must be in Active list (or create new)
        dest_options = [l for l in active_leagues_list if l != from_league]
        to_league = c2.selectbox("Move Matches TO:", dest_options)
        
        if st.button(f"⚠️ Merge '{from_league}' -> '{to_league}'"):
            if from_league and to_league:
                count = df_matches[df_matches['league'].astype(str) == from_league].shape[0]
                supabase.table("matches").update({"league": to_league}).eq("league", from_league).execute()
                st.success(f"✅ Moved {count} matches! Now run 'Recalculate League History' below.")
                time.sleep(2)
                st.rerun()
    
    st.divider()
    
    st.subheader("📅 Bulk Match Date Editor")
    with st.form("bulk_date_edit"):
        c1, c2, c3 = st.columns(3)
        start_id = c1.number_input("Start Match ID", min_value=0)
        end_id = c2.number_input("End Match ID", min_value=0)
        new_date = c3.date_input("New Date")
        
        if st.form_submit_button("Update Match Dates"):
            if start_id > 0 and end_id >= start_id:
                id_list = list(range(start_id, end_id + 1))
                try:
                    supabase.table("matches").update({"date": str(new_date)}).in_("id", id_list).execute()
                    st.success(f"✅ Updated {len(id_list)} matches to {new_date}!")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Invalid ID Range")

    st.divider()
    st.subheader("🔄 Recalculate League History")
    league_options = ["ALL (Full System Reset)"] + sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else ["ALL (Full System Reset)"]
    target_reset = st.selectbox("Select League to Recalculate", league_options)
    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching numbers..."):
            all_players = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute().data
            all_matches = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=False).execute().data
            p_map = {p['id']: {'r': float(p['starting_rating']), 'w': 0, 'l': 0, 'mp': 0, 'name': p['name']} for p in all_players}
            island_map = {}
            matches_to_update = []
            for m in all_matches:
                if target_reset != "ALL (Full System Reset)" and m['league'] != target_reset: continue
                p1, p2, p3, p4 = m['t1_p1'], m['t1_p2'], m['t2_p1'], m['t2_p2']
                s1, s2 = m['score_t1'], m['score_t2']
                league = m['league']
                is_popup = m.get('match_type') == 'PopUp'
                def safe_get_r(pid): return 1200.0 if pid is None else p_map[pid]['r']
                if target_reset == "ALL (Full System Reset)":
                    ro1, ro2, ro3, ro4 = safe_get_r(p1), safe_get_r(p2), safe_get_r(p3), safe_get_r(p4)
                    do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2)
                    delta_to_save = do1 if s1 > s2 else do2
                    matches_to_update.append({'id': m['id'], 'elo_delta': delta_to_save})
                    win = s1 > s2
                    for pid, delta, is_win in [(p1, do1, win), (p2, do1, win), (p3, do2, not win), (p4, do2, not win)]:
                        if pid is None: continue
                        p_map[pid]['r'] += delta
                        p_map[pid]['mp'] += 1
                        if is_win: p_map[pid]['w'] += 1
                        else: p_map[pid]['l'] += 1
                if not is_popup:
                    def safe_get_i(pid, lg):
                        if pid is None: return 1200.0
                        if (pid, lg) not in island_map: island_map[(pid, lg)] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        return island_map[(pid, lg)]['r']
                    ri1, ri2, ri3, ri4 = safe_get_i(p1, league), safe_get_i(p2, league), safe_get_i(p3, league), safe_get_i(p4, league)
                    di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2)
                    win = s1 > s2
                    for pid, delta, is_win in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                        if pid is None: continue
                        k = (pid, league)
                        if k not in island_map: island_map[k] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        island_map[k]['r'] += delta
                        island_map[k]['mp'] += 1
                        if is_win: island_map[k]['w'] += 1
                        else: island_map[k]['l'] += 1
            if target_reset == "ALL (Full System Reset)":
                for pid, stats in p_map.items(): supabase.table("players").update({"rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']}).eq("id", pid).execute()
            if target_reset != "ALL (Full System Reset)": supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", target_reset).execute()
            else: supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()
            new_islands = [{"club_id": CLUB_ID, "player_id": pid, "league_name": lg, "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']} for (pid, lg), stats in island_map.items()]
            if new_islands:
                for i in range(0, len(new_islands), 1000): supabase.table("league_ratings").insert(new_islands[i:i+1000]).execute()
            progress_bar = st.progress(0)
            for idx, update in enumerate(matches_to_update):
                supabase.table("matches").update({"elo_delta": update['elo_delta']}).eq("id", update['id']).execute()
                progress_bar.progress((idx + 1) / len(matches_to_update))
            st.success(f"✅ Replayed {len(all_matches)} matches!"); time.sleep(2); st.rerun()

elif sel == "📘 Admin Guide":
    st.header("📘 Admin Guide")
    st.markdown("""
    ### ⚙️ Admin Tools
    * **League Merger:** Use this to merge "Verified Men's 4.0 -- 2026" into "Verified Men's 4.0" to fix the 0 Weeks bug.
    * **Bulk Date Editor:** Fix "Everything Happened Today" issues.
    """)

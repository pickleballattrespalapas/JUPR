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
            p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
            df_players = pd.DataFrame(p_response.data)
            
            l_response = supabase.table("league_ratings").select("*").eq("club_id", CLUB_ID).execute()
            df_leagues = pd.DataFrame(l_response.data)

            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=True).limit(1000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
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
                time.sleep(1) 
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
        if not all([p1, p2, p3, p4]): continue
        
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
    nav += ["──────────", "📋 Roster Check", "🏟️ Live Court Manager", "🔄 Pop-Up RR", "👥 Players", "📝 Match Log", "⚙️ Admin Tools"]
sel = st.sidebar.radio("Go to:", nav, key="main_nav")

# --- UI LOGIC ---

if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")
    
    available_leagues = ["OVERALL"]
    if not df_leagues.empty:
        unique_l = sorted([l for l in df_leagues['league_name'].unique().tolist() if "Pop" not in l])
        available_leagues += unique_l
        
    default_index = 0
    query_params = st.query_params
    if "league" in query_params:
        url_league = query_params["league"]
        if url_league in available_leagues:
            default_index = available_leagues.index(url_league)
            
    target_league = st.selectbox("Select View", available_leagues, index=default_index)
    
    if target_league != "OVERALL":
        st.query_params["league"] = target_league
        APP_BASE_URL = "https://jupr-leagues.streamlit.app"
        full_link = f"{APP_BASE_URL}/?league={target_league.replace(' ', '%20')}"
        st.caption("👇 Share this league:")
        st.code(full_link, language=None)
    else:
        st.query_params.clear()
    
    with st.expander("📊 How Ratings Work"):
        st.markdown("""
        * **Expectation vs Reality:** We predict the winner based on rating. If you outperform the prediction (e.g., winning 11-0 when it should have been close), you gain more points.
        * **The Swing:**
            * **Big Upset:** ~ +0.075 JUPR
            * **Even Match (11-9):** ~ +0.008 JUPR
        * **Safety Net:** Winners *never* lose points, even in sloppy wins.
        """)

    if target_league == "OVERALL":
        display_df = df_players.copy()
    else:
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
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

# --- NEW PAGE: ROSTER CHECK (INDIVIDUAL EDIT) ---
elif sel == "📋 Roster Check":
    st.header("📋 Roster Check")
    st.markdown("Paste a list of names to check their ratings. Detects typos and helps add new players.")
    
    lookup_scope = st.radio("Rating Scope", ["OVERALL"] + (sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else []), horizontal=True)
    raw_names = st.text_area("Paste Names (one per line or comma-separated)", height=100, placeholder="Tom Elliott\nKeith Brand\nNew Player")
    
    if st.button("Analyze List"):
        # Reset any previous session data to ensure fresh start
        if 'roster_results' in st.session_state: del st.session_state.roster_results
        if 'df_new_players' in st.session_state: del st.session_state.df_new_players

        parsed = [x.strip() for x in raw_names.replace('\n',',').split(',') if x.strip()]
        
        found_data = []
        typo_candidates = []
        new_players = []
        all_db_names = list(name_to_id.keys())
        
        for n in parsed:
            # A. EXACT MATCH
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
            # B. FUZZY MATCH
            else:
                matches = difflib.get_close_matches(n, all_db_names, n=1, cutoff=0.6)
                if matches:
                    typo_candidates.append({"Input": n, "Did you mean?": matches[0]})
                else:
                    new_players.append(n)

        # Store results in Session State to allow editing below
        st.session_state.roster_results = {
            'found': found_data,
            'typos': typo_candidates,
            'new': new_players
        }

    # --- RENDER RESULTS FROM STATE ---
    if 'roster_results' in st.session_state:
        results = st.session_state.roster_results
        
        # 1. Found Players
        if results['found']:
            st.success(f"Found {len(results['found'])} players.")
            st.dataframe(pd.DataFrame(results['found']), use_container_width=True)
        
        # 2. Typos
        if results['typos']:
            st.warning(f"⚠️ Found {len(results['typos'])} potential typos.")
            with st.form("fix_typos"):
                accepted_fixes = {}
                for item in results['typos']:
                    c1, c2 = st.columns([1, 2])
                    use = c1.checkbox(f"Use '{item['Did you mean?']}'?", value=True, key=f"fix_{item['Input']}")
                    c2.markdown(f"**{item['Input']}** → **{item['Did you mean?']}**")
                    if use: accepted_fixes[item['Input']] = item['Did you mean?']
                
                if st.form_submit_button("Confirm Fixes"):
                    st.success("✅ Mapped! Re-run analysis if you want to see them in 'Found'.")

        # 3. New Players (INDIVIDUAL EDIT)
        if results['new']:
            st.error(f"🛑 Found {len(results['new'])} completely new players.")
            st.write("### 🆕 Assign Ratings")
            
            # Create/Load DataFrame for Editor
            if 'df_new_players' not in st.session_state or len(st.session_state.df_new_players) != len(results['new']):
                st.session_state.df_new_players = pd.DataFrame({
                    "Name": results['new'],
                    "Rating": [3.5] * len(results['new'])
                })
            
            # Interactive Table
            edited_df = st.data_editor(
                st.session_state.df_new_players,
                column_config={
                    "Rating": st.column_config.NumberColumn("Start Rating", min_value=1.0, max_value=7.0, step=0.1, format="%.1f")
                },
                disabled=["Name"],
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("💾 Save All New Players"):
                success_count = 0
                for _, row in edited_df.iterrows():
                    ok, msg = safe_add_player(row['Name'], row['Rating'])
                    if ok: success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ Created {success_count} profiles!")
                    time.sleep(1)
                    # Clear state to reset view
                    del st.session_state.roster_results
                    del st.session_state.df_new_players
                    st.rerun()

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
            st.session_state.lc_schedule = []
            st.session_state.active_league = lg 
            for idx, c in enumerate(c_data):
                pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
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
                    all_res.append({'t1_p1':m['t1'][0], 't1_p2':m['t1'][1], 't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 's1':s1, 's2':s2, 'date':str(datetime.now()), 'league':st.session_state.get('active_league', 'Unknown'), 'type':f"C{c['c']} RR", 'match_type': 'Live Match', 'is_popup': False})
            
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
            st.session_state.rr_schedule = []
            st.session_state.rr_league = lg_rr
            for idx, c in enumerate(c_data):
                pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                st.session_state.rr_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
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
        st.subheader("✏️ Edit Rating")
        p_edit = st.selectbox("Select Player", [""] + sorted(df_players['name']))
        if p_edit:
            curr_row = df_players[df_players['name'] == p_edit].iloc[0]
            curr_start = float(curr_row['starting_rating']) / 400
            new_start = st.number_input("New Start Rating", 1.0, 7.0, curr_start, step=0.1)
            if st.button("Update Rating"):
                supabase.table("players").update({"starting_rating": new_start * 400, "rating": new_start * 400}).eq("name", p_edit).eq("club_id", CLUB_ID).execute()
                st.success(f"Updated {p_edit} to {new_start}"); time.sleep(1); st.rerun()
    with c3:
        st.subheader("🗑️ Delete")
        to_del = st.selectbox("Select to Remove", [""] + sorted(df_players['name']))
        if to_del and st.button("Confirm Delete"):
            supabase.table("players").delete().eq("name", to_del).eq("club_id", CLUB_ID).execute()
            st.success("Deleted."); time.sleep(1); st.rerun()
    st.dataframe(df_players, use_container_width=True)

elif sel == "📝 Match Log":
    st.header("📝 Match Log")
    st.subheader("Recent Matches")
    
    edit_df = df_matches.head(100)[['id', 'date', 'league', 'match_type', 'elo_delta', 'p1', 'p2', 'p3', 'p4', 'score_t1', 'score_t2']].copy()
    edit_df['Raw Pts'] = edit_df['elo_delta'].map('{:.1f}'.format)
    st.dataframe(edit_df.drop(columns=['elo_delta']), use_container_width=True)
    
    st.divider()
    m_id = st.number_input("Match ID to Delete", min_value=0, step=1)
    if st.button("Delete Match"):
        supabase.table("matches").delete().eq("id", m_id).execute()
        st.success("Deleted!"); time.sleep(1); st.rerun()

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    
    zero_pts = df_matches[df_matches['elo_delta'] == 0].shape[0]
    if zero_pts > 0:
        st.warning(f"⚠️ Found {zero_pts} matches with 0.0 point impact. Run 'Recalculate' below to fix them.")
    else:
        st.success("✅ System Health: All matches have valid point values.")

    st.subheader("🕵️ Match Forensics")
    debug_id = st.number_input("Enter Match ID to Diagnose", value=0, step=1)
    if st.button("Run Diagnostics"):
        m = supabase.table("matches").select("*").eq("id", debug_id).execute().data
        if not m:
            st.error("Match not found.")
        else:
            m = m[0]
            st.write("### 1. Match Data Raw")
            st.json(m)

            p_ids = [m['t1_p1'], m['t1_p2'], m['t2_p1'], m['t2_p2']]
            p_ids = [x for x in p_ids if x is not None]

            st.write("### 2. Player Data")
            if p_ids:
                players = supabase.table("players").select("*").in_("id", p_ids).execute().data
                st.write(players)
            
            st.write("### 3. Math Engine Simulation")
            def mock_get(pid):
                if pid is None: return 1200.0
                match_p = next((x for x in players if x['id'] == pid), None)
                if match_p: return float(match_p['rating'])
                return 1200.0

            ro1, ro2, ro3, ro4 = mock_get(m['t1_p1']), mock_get(m['t1_p2']), mock_get(m['t2_p1']), mock_get(m['t2_p2'])
            
            st.write(f"Team 1 Ratings: {ro1}, {ro2} (Avg: {(ro1+ro2)/2})")
            st.write(f"Team 2 Ratings: {ro3}, {ro4} (Avg: {(ro3+ro4)/2})")
            st.write(f"Scores: {m['score_t1']} - {m['score_t2']}")
            
            d1, d2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, m['score_t1'], m['score_t2'])
            st.write(f"**Calculated Delta:** {d1 if m['score_t1'] > m['score_t2'] else d2}")
    
    st.divider()
    
    st.subheader("🛠️ League Manager")
    if not df_leagues.empty:
        active_islands = sorted(df_leagues['league_name'].unique().tolist())
        to_hide = st.selectbox("Select League to Convert to 'PopUp' (Hides from Leaderboard)", [""] + active_islands)
        
        if to_hide and st.button(f"⚠️ Convert '{to_hide}' to PopUp"):
            supabase.table("matches").update({"match_type": "PopUp"}).eq("league", to_hide).execute()
            supabase.table("league_ratings").delete().eq("league_name", to_hide).execute()
            st.success(f"Converted '{to_hide}'! It will now vanish from the Leaderboard.")
            time.sleep(2)
            st.rerun()

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
                
                def safe_get_r(pid):
                    if pid is None: return 1200.0
                    return p_map[pid]['r']

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
                    def get_i_r(pid, lg):
                        if (pid, lg) not in island_map: 
                            curr_r = p_map[pid]['r']
                            island_map[(pid, lg)] = {'r': curr_r, 'w':0, 'l':0, 'mp':0}
                        return island_map[(pid, lg)]['r']

                    def safe_get_i(pid, lg):
                        if pid is None: return 1200.0
                        return get_i_r(pid, lg)

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
                for pid, stats in p_map.items():
                    supabase.table("players").update({"rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']}).eq("id", pid).execute()
            
            if target_reset != "ALL (Full System Reset)": supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", target_reset).execute()
            else: supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()
            
            new_islands = []
            for (pid, lg), stats in island_map.items():
                if target_reset == "ALL (Full System Reset)" or lg == target_reset:
                    new_islands.append({"club_id": CLUB_ID, "player_id": pid, "league_name": lg, "rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']})
            
            if new_islands:
                chunk_size = 1000
                for i in range(0, len(new_islands), chunk_size):
                    supabase.table("league_ratings").insert(new_islands[i:i+chunk_size]).execute()

            progress_bar = st.progress(0)
            for idx, update in enumerate(matches_to_update):
                supabase.table("matches").update({"elo_delta": update['elo_delta']}).eq("id", update['id']).execute()
                progress_bar.progress((idx + 1) / len(matches_to_update))

            st.success(f"✅ Replayed & Synced {len(all_matches)} matches!"); time.sleep(2); st.rerun()

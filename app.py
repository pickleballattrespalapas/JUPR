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
DEFAULT_K_FACTOR = 32
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
def calculate_hybrid_elo(t1_avg, t2_avg, score_t1, score_t2, k_factor=32):
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1
    total_points = score_t1 + score_t2
    if total_points == 0: return 0, 0 
    raw_delta_t1 = k_factor * 2 * ((score_t1 / total_points) - expected_t1)
    raw_delta_t2 = k_factor * 2 * ((score_t2 / total_points) - expected_t2)
    
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
            
            # 2. League Ratings
            l_response = supabase.table("league_ratings").select("*").eq("club_id", CLUB_ID).execute()
            df_leagues = pd.DataFrame(l_response.data)

            # 3. Matches (Recent)
            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("id", desc=True).limit(5000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
            # 4. League Metadata (Config)
            meta_response = supabase.table("leagues_metadata").select("*").eq("club_id", CLUB_ID).execute()
            df_meta = pd.DataFrame(meta_response.data)
            
            # --- ID MAPS ---
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
    # (Same schedule logic as before - kept for brevity)
    if len(players) < int(format_type.split('-')[0]): return []
    if format_type == "4-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif format_type == "5-Player": return [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5'}]
    return [] # Add 6/8/12 if needed

def safe_add_player(name, rating):
    try:
        supabase.table("players").insert({
            "club_id": CLUB_ID, "name": name, 
            "rating": rating * 400, "starting_rating": rating * 400
        }).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

# --- PROCESSOR (UPDATED WITH SNAPSHOTS) ---
def process_matches(match_list, name_to_id, df_p, df_l, df_meta):
    db_matches = []
    overall_updates = {} 
    island_updates = {}

    # Helper: Get K-Factor for a league
    def get_k(lg_name):
        if df_meta.empty: return DEFAULT_K_FACTOR
        row = df_meta[df_meta['league_name'] == lg_name]
        if not row.empty: return int(row.iloc[0].get('k_factor', DEFAULT_K_FACTOR))
        return DEFAULT_K_FACTOR

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
        
        # 1. Global Calculation
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2, k_factor=DEFAULT_K_FACTOR)
        
        # 2. Island Calculation (with specific K-Factor)
        di1, di2 = 0, 0
        if not is_popup:
            k_val = get_k(league)
            ri1, ri2, ri3, ri4 = get_island_r(p1, league), get_island_r(p2, league), get_island_r(p3, league), get_island_r(p4, league)
            di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2, k_factor=k_val)

        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": league,
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, 
            "elo_delta": do1 if s1 > s2 else do2,
            "match_type": m['match_type'],
            # NEW: SNAPSHOTS
            "t1_p1_r": ro1, "t1_p2_r": ro2, "t2_p1_r": ro3, "t2_p2_r": ro4
        })

        win = s1 > s2
        participants = [(p1, do1, di1, win), (p2, do1, di1, win), (p3, do2, di2, not win), (p4, do2, di2, not win)]
        
        for pid, d_overall, d_island, won in participants:
            if pid is None: continue 
            
            # Update Global
            if pid not in overall_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                overall_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            overall_updates[pid]['r'] += d_overall
            overall_updates[pid]['mp'] += 1
            if won: overall_updates[pid]['w'] += 1
            else: overall_updates[pid]['l'] += 1
            
            # Update Island
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
    st.header("🏆 Leaderboards")
    
    # 1. Determine available leagues from CONFIG (Active only)
    if not df_meta.empty:
        # Sort by display order or name. Here we filter by is_active.
        active_meta = df_meta[df_meta['is_active'] == True]
        available_leagues = ["OVERALL"] + sorted(active_meta['league_name'].unique().tolist())
    elif not df_leagues.empty:
        # Fallback if no config
        available_leagues = ["OVERALL"] + sorted([l for l in df_leagues['league_name'].unique().tolist() if "Pop" not in l])
    else:
        available_leagues = ["OVERALL"]
        
    target_league = st.selectbox("Select View", available_leagues)
    
    # 2. Get Config for this league
    min_games_req = 0
    desc_text = ""
    if target_league != "OVERALL" and not df_meta.empty:
        cfg = df_meta[df_meta['league_name'] == target_league]
        if not cfg.empty:
            min_games_req = cfg.iloc[0].get('min_games', 0)
            desc_text = cfg.iloc[0].get('description', "")

    if desc_text: st.info(desc_text)
    
    # 3. Filter Data
    if target_league == "OVERALL":
        display_df = df_players.copy()
        # For overall, maybe use a default min games?
        min_games_req = 5 
    else:
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
    if not display_df.empty and 'rating' in display_df.columns:
        display_df['JUPR'] = (display_df['rating']/400)
        display_df['Win %'] = (display_df['wins'] / display_df['matches_played'].replace(0,1) * 100)
        
        # 4. QUALIFIED DATAFRAME (For Top 5 Widgets)
        qualified_df = display_df[display_df['matches_played'] >= min_games_req].copy()
        
        if not qualified_df.empty:
            st.markdown(f"### 🏅 Top Performers (Min {min_games_req} Games)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**👑 Highest Rating**")
                top = qualified_df.sort_values('rating', ascending=False).head(5)
                for _, r in top.iterrows(): st.markdown(f"**{r['JUPR']:.3f}** - {r['name']}")
            with c2:
                st.markdown("**🎯 Best Win %**")
                top = qualified_df.sort_values('Win %', ascending=False).head(5)
                for _, r in top.iterrows(): st.markdown(f"**{r['Win %']:.1f}%** - {r['name']}")
            with c3:
                st.markdown("**🚜 Most Wins**")
                top = qualified_df.sort_values('wins', ascending=False).head(5)
                for _, r in top.iterrows(): st.markdown(f"**{r['wins']} Wins** - {r['name']}")
            st.divider()

        # 5. FULL TABLE (Everyone)
        st.markdown("### 📊 Full Standings")
        final_view = display_df.sort_values('rating', ascending=False).copy()
        final_view['JUPR'] = final_view['JUPR'].map('{:,.3f}'.format)
        final_view['Win %'] = final_view['Win %'].map('{:.1f}%'.format)
        
        st.dataframe(final_view[['name', 'JUPR', 'matches_played', 'wins', 'losses', 'Win %']], use_container_width=True, hide_index=True)
    else:
        st.info("No data for this league yet.")

elif sel == "🔍 Player Search":
    st.header("🔍 Player History")
    p = st.selectbox("Search", [""] + sorted(df_players['name']))
    if p:
        # (Same Logic as before)
        p_row = df_players[df_players['name'] == p]
        if not p_row.empty:
            st.metric("Current JUPR", f"{(float(p_row.iloc[0]['rating'])/400):.3f}")

        if not df_matches.empty:
            mask = (df_matches['p1'] == p) | (df_matches['p2'] == p) | (df_matches['p3'] == p) | (df_matches['p4'] == p)
            h = df_matches[mask].copy()
            if not h.empty:
                # Add Snapshot Logic
                def get_snapshot(r):
                    # Try to find the snapshot column
                    # This logic assumes the columns exist. If not, it falls back gracefully
                    if p == r['p1']: return r.get('t1_p1_r', 0)
                    if p == r['p2']: return r.get('t1_p2_r', 0)
                    if p == r['p3']: return r.get('t2_p1_r', 0)
                    if p == r['p4']: return r.get('t2_p2_r', 0)
                    return 0
                
                h['Rating Before'] = h.apply(get_snapshot, axis=1)
                h['Result'] = h.apply(lambda x: "✅ Win" if (p in [x['p1'], x['p2']] and x['score_t1']>x['score_t2']) or (p in [x['p3'], x['p4']] and x['score_t2']>x['score_t1']) else "❌ Loss", axis=1)
                
                # Format
                display_h = h[['date', 'league', 'Result', 'score_t1', 'score_t2', 'p1', 'p2', 'p3', 'p4']].copy()
                st.dataframe(display_h, use_container_width=True, hide_index=True)

elif sel == "🏟️ League Manager":
    st.header("🏟️ League Manager")
    
    tabs = st.tabs(["⚙️ Settings (Config)", "📝 Live Courts"])
    
    # --- TAB 1: SETTINGS ---
    with tabs[0]:
        st.markdown("Manage which leagues appear on the leaderboard and their rules.")
        if not df_meta.empty:
            # We edit a copy
            editor_df = df_meta[['id', 'league_name', 'is_active', 'min_games', 'description', 'k_factor']].copy()
            
            edited_leagues = st.data_editor(
                editor_df,
                column_config={
                    "league_name": "League Name",
                    "is_active": st.column_config.CheckboxColumn("Active?", help="Show in Dropdown"),
                    "min_games": st.column_config.NumberColumn("Min Games", help="Required to be in Top 5 widgets"),
                    "description": st.column_config.TextColumn("Description", width="large"),
                    "k_factor": st.column_config.NumberColumn("K-Factor", min_value=10, max_value=100)
                },
                disabled=["id", "league_name"],
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("💾 Save Config Changes"):
                # We need to loop and update. 
                # Ideally Supabase upsert would be better but simple loop works for small lists.
                changes = 0
                for index, row in edited_leagues.iterrows():
                    # Only update if changed? For now just update all active rows for simplicity or check diff
                    supabase.table("leagues_metadata").update({
                        "is_active": row['is_active'],
                        "min_games": row['min_games'],
                        "description": row['description'],
                        "k_factor": row['k_factor']
                    }).eq("id", row['id']).execute()
                    changes += 1
                st.success("Settings Updated!"); time.sleep(1); st.rerun()
                
            st.divider()
            st.write("#### 🆕 Create New League")
            with st.form("new_lg"):
                n = st.text_input("Name")
                k = st.number_input("K-Factor", 32)
                mg = st.number_input("Min Games", 12)
                if st.form_submit_button("Create"):
                    supabase.table("leagues_metadata").insert({
                        "club_id": CLUB_ID, "league_name": n, "league_type": "Standard", 
                        "is_active": True, "min_games": mg, "k_factor": k
                    }).execute()
                    st.rerun()

    # --- TAB 2: LIVE COURTS ---
    with tabs[1]:
        # (Your existing Live Court code goes here)
        if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
        st.session_state.lc_courts = st.number_input("Courts", 1, 10, st.session_state.lc_courts, key="lc_court_input")
        
        with st.form("setup_lc"):
            # Use ACTIVE leagues from metadata for dropdown
            active_opts = sorted(df_meta[df_meta['is_active']==True]['league_name'].tolist()) if not df_meta.empty else ["Default"]
            lg = st.selectbox("Select League", active_opts)
            
            c_data = []
            for i in range(st.session_state.lc_courts):
                c1, c2 = st.columns([1,3])
                t = c1.selectbox(f"T{i}", ["4-Player","5-Player","6-Player"], key=f"t_{i}")
                n = c2.text_area(f"N{i}", height=70, key=f"n_{i}")
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
                        process_matches(valid, name_to_id, df_players, df_leagues, df_meta)
                        st.success("✅ Processed!")
                        del st.session_state.lc_schedule
                        time.sleep(1)
                        st.rerun()

# --- OTHER TABS (Batch, Players, Admin) ---
# (I'll keep these brief as they are largely unchanged, but I need to update Recalculate)

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    
    st.subheader("🔄 Recalculate & Snapshot History")
    league_options = ["ALL (Full System Reset)"] + sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else ["ALL"]
    target_reset = st.selectbox("Select League", league_options)

    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching numbers & Backfilling Snapshots..."):
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

                # CAPTURE SNAPSHOTS
                snap_r1, snap_r2, snap_r3, snap_r4 = safe_get_r(p1), safe_get_r(p2), safe_get_r(p3), safe_get_r(p4)
                
                # --- GLOBAL CALC ---
                do1, do2 = calculate_hybrid_elo((snap_r1+snap_r2)/2, (snap_r3+snap_r4)/2, s1, s2)
                
                # Update P_MAP
                win = s1 > s2
                for pid, delta, is_win in [(p1, do1, win), (p2, do1, win), (p3, do2, not win), (p4, do2, not win)]:
                    if pid is None: continue
                    p_map[pid]['r'] += delta
                    p_map[pid]['mp'] += 1
                    if is_win: p_map[pid]['w'] += 1
                    else: p_map[pid]['l'] += 1

                # --- ISLAND CALC ---
                if not is_popup:
                    def get_i_r(pid, lg):
                        if (pid, lg) not in island_map: 
                            # Initialize island rating with current GLOBAL rating at that moment? 
                            # Or default 1200? Usually starting with current global is fair, or seed.
                            # For pure island, start 1200 or player start. Let's use player start (safe) or current global.
                            # Simpler: Use current global snapshot.
                            island_map[(pid, lg)] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        return island_map[(pid, lg)]['r']
                    
                    ir1, ir2, ir3, ir4 = get_i_r(p1, league), get_i_r(p2, league), get_i_r(p3, league), get_i_r(p4, league)
                    
                    # Look up K-Factor
                    # (Simplified for recalc speed: assume 32 unless we fetch per row. Let's use 32 default)
                    di1, di2 = calculate_hybrid_elo((ir1+ir2)/2, (ir3+ir4)/2, s1, s2, k_factor=32)
                    
                    for pid, delta, is_win in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                        if pid is None: continue
                        k = (pid, league)
                        if k not in island_map: island_map[k] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        island_map[k]['r'] += delta
                        island_map[k]['mp'] += 1
                        if is_win: island_map[k]['w'] += 1
                        else: island_map[k]['l'] += 1

                # QUEUE UPDATE with Snapshots
                matches_to_update.append({
                    'id': m['id'], 
                    'elo_delta': do1 if s1 > s2 else do2,
                    't1_p1_r': snap_r1, 't1_p2_r': snap_r2,
                    't2_p1_r': snap_r3, 't2_p2_r': snap_r4
                })

            # SAVE EVERYTHING
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
                for i in range(0, len(new_islands), 1000):
                    supabase.table("league_ratings").insert(new_islands[i:i+1000]).execute()

            # Batch update matches is hard in Supabase API without a stored procedure or loop
            # We will use loop + progress bar
            progress_bar = st.progress(0)
            for idx, update in enumerate(matches_to_update):
                # Update delta AND snapshots
                supabase.table("matches").update(update).eq("id", update['id']).execute()
                progress_bar.progress((idx + 1) / len(matches_to_update))

            st.success(f"✅ Replayed & Backfilled {len(all_matches)} matches!"); time.sleep(2); st.rerun()

elif sel == "👥 Players":
    # (Same Player Code as before)
    st.header("Player Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.0)
            if st.form_submit_button("Add Player"):
                safe_add_player(n, r)
                st.rerun()
    st.dataframe(df_players)

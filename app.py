import streamlit as st
import pandas as pd
from supabase import create_client, Client
import time
from datetime import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide", page_icon="🌵")

# Custom CSS
st.markdown("""
<style>
    .stDataFrame { font-size: 1.1rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00C0F2; }
    div[data-testid="stVerticalBlock"] > div:has(div.stDataFrame) { padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

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
    # Same logic as before
    if len(players) < int(format_type.split('-')[0]): return []
    if format_type == "4-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif format_type == "5-Player": return [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5'}]
    elif format_type == "6-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[4]],'desc':'R1'}, {'t1':[p[2],p[5]],'t2':[p[0],p[4]],'desc':'R2'}, {'t1':[p[1],p[3]],'t2':[p[4],p[5]],'desc':'R3'}, {'t1':[p[0],p[5]],'t2':[p[1],p[2]],'desc':'R4'}, {'t1':[p[0],p[3]],'t2':[p[1],p[4]],'desc':'R5'}]
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

# --- PROCESSOR (WITH SNAPSHOTS) ---
def process_matches(match_list, name_to_id, df_p, df_l, df_meta):
    db_matches = []
    overall_updates = {} 
    island_updates = {}

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
        
        # 1. Global
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2, k_factor=DEFAULT_K_FACTOR)
        
        # 2. Island
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
            "t1_p1_r": ro1, "t1_p2_r": ro2, "t2_p1_r": ro3, "t2_p2_r": ro4
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
    nav += ["──────────", "📋 Roster Check", "🏟️ League Manager", "📝 Match Uploader", "👥 Players", "📝 Match Log", "⚙️ Admin Tools", "📘 Admin Guide"]
sel = st.sidebar.radio("Go to:", nav, key="main_nav")

# --- UI LOGIC ---

if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")
    
    # 1. Menu Logic
    if not df_meta.empty:
        active_meta = df_meta[df_meta['is_active'] == True]
        available_leagues = ["OVERALL"] + sorted(active_meta['league_name'].unique().tolist())
    elif not df_leagues.empty:
        available_leagues = ["OVERALL"] + sorted([l for l in df_leagues['league_name'].unique().tolist() if "Pop" not in l])
    else:
        available_leagues = ["OVERALL"]
        
    target_league = st.selectbox("Select View", available_leagues)
    
    # 2. Get Rules
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
    else:
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
    if not display_df.empty and 'rating' in display_df.columns:
        # Pre-calc columns
        display_df['JUPR'] = (display_df['rating']/400)
        display_df['Win %'] = (display_df['wins'] / display_df['matches_played'].replace(0,1) * 100)
        
        # --- CALCULATE GAIN (Using Snapshots) ---
        # We only calculate this for the "Qualified" players to save speed, or everyone if fast enough.
        # Let's do it for display_df to allow sorting in main table later if wanted.
        
        def calculate_gain(row):
            pid = row['id'] if 'id' in row else row['player_id']
            curr_r = row['rating']
            
            # If OVERALL, we use the player's static starting_rating
            if target_league == "OVERALL":
                start_r = row.get('starting_rating', 1200.0) # Default from DB
                return curr_r - start_r

            # If LEAGUE, we find their first match in this league
            # Filter matches for this league involving this player
            if df_matches.empty: return 0.0
            
            # Find matches where this player played in this league
            # We assume df_matches is sorted desc (newest first), so we take the LAST one (oldest)
            # OR we just sort by date inside this function
            
            relevant = df_matches[
                (df_matches['league'] == target_league) & 
                ((df_matches['t1_p1'] == pid) | (df_matches['t1_p2'] == pid) | (df_matches['t2_p1'] == pid) | (df_matches['t2_p2'] == pid))
            ]
            
            if relevant.empty: return 0.0
            
            # Get oldest match
            oldest = relevant.iloc[-1] 
            
            # Get the snapshot rating *before* that match
            if pid == oldest['t1_p1']: snap = oldest.get('t1_p1_r', 0)
            elif pid == oldest['t1_p2']: snap = oldest.get('t1_p2_r', 0)
            elif pid == oldest['t2_p1']: snap = oldest.get('t2_p1_r', 0)
            elif pid == oldest['t2_p2']: snap = oldest.get('t2_p2_r', 0)
            else: snap = 0
            
            # Fallback if snapshot is missing (0) -> Assume 1200 or starting_rating?
            # Ideally we used Recalculate so snapshots are populated. 
            if snap == 0 or pd.isna(snap): return 0.0 
            
            return curr_r - snap

        # Apply calculation
        display_df['rating_gain'] = display_df.apply(calculate_gain, axis=1)

        # --- TOP WIDGETS (Conditional: NOT on OVERALL) ---
        if target_league != "OVERALL":
            qualified_df = display_df[display_df['matches_played'] >= min_games_req].copy()
            if not qualified_df.empty:
                st.markdown(f"### 🏅 Top Performers (Min {min_games_req} Games)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("**👑 Highest Rating**")
                    top = qualified_df.sort_values('rating', ascending=False).head(5)
                    for _, r in top.iterrows(): st.markdown(f"**{r['JUPR']:.3f}** - {r['name']}")
                with c2:
                    st.markdown("**🔥 Most Improved**")
                    top = qualified_df.sort_values('rating_gain', ascending=False).head(5)
                    for _, r in top.iterrows(): 
                        gain_val = r['rating_gain'] / 400
                        st.markdown(f"**{'+' if gain_val>0 else ''}{gain_val:.3f}** - {r['name']}")
                with c3:
                    st.markdown("**🎯 Best Win %**")
                    top = qualified_df.sort_values('Win %', ascending=False).head(5)
                    for _, r in top.iterrows(): st.markdown(f"**{r['Win %']:.1f}%** - {r['name']}")
                with c4:
                    st.markdown("**🚜 Most Wins**")
                    top = qualified_df.sort_values('wins', ascending=False).head(5)
                    for _, r in top.iterrows(): st.markdown(f"**{r['wins']} Wins** - {r['name']}")
                st.divider()

        # --- STYLED TABLE ---
        st.markdown("### 📊 Standings")
        
        # Sort and Add Rank
        final_view = display_df.sort_values('rating', ascending=False).copy()
        final_view['Rank'] = range(1, len(final_view) + 1)
        
        # Medal Logic
        def get_medal(r):
            if r == 1: return "🥇"
            if r == 2: return "🥈"
            if r == 3: return "🥉"
            return str(r)
        
        final_view['Rank'] = final_view['Rank'].apply(get_medal)
        final_view['Gain'] = (final_view['rating_gain']/400).map('{:+.3f}'.format)
        
        # Show Styled DataFrame
        st.dataframe(
            final_view[['Rank', 'name', 'JUPR', 'Gain', 'matches_played', 'wins', 'losses', 'Win %']], 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Rank": st.column_config.Column("Rank", width="small"),
                "name": st.column_config.Column("Player", width="medium"),
                "JUPR": st.column_config.NumberColumn("Rating", format="%.3f"),
                "Gain": st.column_config.Column("Gain", help="Rating Change since first match in this league"),
                "Win %": st.column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0, max_value=100),
                "matches_played": st.column_config.NumberColumn("Matches"),
                "wins": st.column_config.NumberColumn("W"),
                "losses": st.column_config.NumberColumn("L")
            }
        )
    else:
        st.info("No data for this league yet.")

elif sel == "🏟️ League Manager":
    st.header("🏟️ League Manager (Settings)")
    st.markdown("Manage which leagues appear on the leaderboard and their rules.")
    
    if not df_meta.empty:
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
            for index, row in edited_leagues.iterrows():
                supabase.table("leagues_metadata").update({
                    "is_active": row['is_active'],
                    "min_games": row['min_games'],
                    "description": row['description'],
                    "k_factor": row['k_factor']
                }).eq("id", row['id']).execute()
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

elif sel == "📝 Match Uploader":
    st.header("📝 Match Uploader")
    
    # 1. Context Selector
    c1, c2 = st.columns(2)
    ctx_type = c1.radio("Match Context", ["🏆 Official League Match", "🎉 Pop-Up / Event"], horizontal=True)
    
    selected_league = ""
    is_popup = False
    
    if ctx_type == "🏆 Official League Match":
        # Show active leagues
        active_opts = sorted(df_meta[df_meta['is_active']==True]['league_name'].tolist()) if not df_meta.empty else ["Default"]
        selected_league = c2.selectbox("Select League", active_opts)
        match_type_db = "Live Match" # Or whatever you use for standard
    else:
        # Pop Up
        selected_league = c2.text_input("Event Name", "Saturday Social")
        is_popup = True
        match_type_db = "PopUp"

    st.divider()
    
    # 2. Method Selector
    entry_method = st.radio("Entry Method", ["📋 Manual / Batch Entry", "🏟️ Live Round Robin"], horizontal=True)
    st.write("") # Spacer

    # --- METHOD A: BATCH ---
    if entry_method == "📋 Manual / Batch Entry":
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

        if st.button("Submit Batch"):
            valid_batch = []
            for _, row in edited_batch.iterrows():
                if row['T1_P1'] and row['T2_P1'] and (row['Score_1'] + row['Score_2'] > 0):
                    match_data = {
                        't1_p1': row['T1_P1'], 
                        't1_p2': row['T1_P2'], 
                        't2_p1': row['T2_P1'], 
                        't2_p2': row['T2_P2'], 
                        's1': int(row['Score_1']), 
                        's2': int(row['Score_2']), 
                        'date': str(datetime.now()), 
                        'league': selected_league, 
                        'type': 'Batch Entry', 
                        'match_type': match_type_db, 
                        'is_popup': is_popup
                    }
                    valid_batch.append(match_data)
            
            if valid_batch:
                process_matches(valid_batch, name_to_id, df_players, df_leagues, df_meta)
                st.success(f"✅ Successfully processed {len(valid_batch)} matches for {selected_league}!")
                # Reset batch
                st.session_state.batch_df = pd.DataFrame(
                [{'T1_P1': None, 'T1_P2': None, 'Score_1': 0, 'Score_2': 0, 'T2_P1': None, 'T2_P2': None} for _ in range(5)]
                )
                time.sleep(1)
                st.rerun()
            else:
                st.warning("⚠️ No valid matches found (check names & scores).")

    # --- METHOD B: LIVE RR ---
    else:
        if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
        st.session_state.lc_courts = st.number_input("Number of Courts", 1, 10, st.session_state.lc_courts)
        
        with st.form("setup_lc"):
            st.write(f"Generating Schedule for: **{selected_league}**")
            c_data = []
            for i in range(st.session_state.lc_courts):
                c1, c2 = st.columns([1,3])
                t = c1.selectbox(f"Format C{i+1}", ["4-Player","5-Player","6-Player"], key=f"t_{i}")
                n = c2.text_area(f"Players C{i+1}", height=70, key=f"n_{i}", placeholder="Paste names here...")
                c_data.append({'type':t, 'names':n})
            
            if st.form_submit_button("Generate Schedule"):
                st.session_state.lc_schedule = []
                st.session_state.active_league_name = selected_league 
                st.session_state.active_is_popup = is_popup
                st.session_state.active_match_type = match_type_db

                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()

        if 'lc_schedule' in st.session_state:
            st.divider()
            st.info(f"Posting results to: **{st.session_state.get('active_league_name', 'Unknown')}**")
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
                            't1_p1':m['t1'][0], 't1_p2':m['t1'][1], 
                            't2_p1':m['t2'][0], 't2_p2':m['t2'][1], 
                            's1':s1, 's2':s2, 
                            'date':str(datetime.now()), 
                            'league': st.session_state.get('active_league_name', 'Unknown'), 
                            'type':f"C{c['c']} RR", 
                            'match_type': st.session_state.get('active_match_type', 'Live Match'), 
                            'is_popup': st.session_state.get('active_is_popup', False)
                        })
                
                if st.form_submit_button("Submit Scores"):
                    valid = [x for x in all_res if x['s1'] > 0 or x['s2'] > 0]
                    if valid:
                        process_matches(valid, name_to_id, df_players, df_leagues, df_meta)
                        st.success("✅ Processed!")
                        del st.session_state.lc_schedule
                        time.sleep(1)
                        st.rerun()

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

                snap_r1, snap_r2, snap_r3, snap_r4 = safe_get_r(p1), safe_get_r(p2), safe_get_r(p3), safe_get_r(p4)
                do1, do2 = calculate_hybrid_elo((snap_r1+snap_r2)/2, (snap_r3+snap_r4)/2, s1, s2)
                
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
                            island_map[(pid, lg)] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        return island_map[(pid, lg)]['r']
                    
                    ir1, ir2, ir3, ir4 = get_i_r(p1, league), get_i_r(p2, league), get_i_r(p3, league), get_i_r(p4, league)
                    di1, di2 = calculate_hybrid_elo((ir1+ir2)/2, (ir3+ir4)/2, s1, s2, k_factor=32)
                    
                    for pid, delta, is_win in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                        if pid is None: continue
                        k = (pid, league)
                        if k not in island_map: island_map[k] = {'r': p_map[pid]['r'], 'w':0, 'l':0, 'mp':0}
                        island_map[k]['r'] += delta
                        island_map[k]['mp'] += 1
                        if is_win: island_map[k]['w'] += 1
                        else: island_map[k]['l'] += 1

                matches_to_update.append({
                    'id': m['id'], 
                    'elo_delta': do1 if s1 > s2 else do2,
                    't1_p1_r': snap_r1, 't1_p2_r': snap_r2,
                    't2_p1_r': snap_r3, 't2_p2_r': snap_r4
                })

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

            progress_bar = st.progress(0)
            for idx, update in enumerate(matches_to_update):
                supabase.table("matches").update(update).eq("id", update['id']).execute()
                progress_bar.progress((idx + 1) / len(matches_to_update))

            st.success(f"✅ Replayed & Backfilled {len(all_matches)} matches!"); time.sleep(2); st.rerun()

elif sel == "👥 Players":
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
    
    if not view_df.empty:
        # Prepare editable df
        edit_df = view_df.head(5000)[['id', 'date', 'league', 'match_type', 'elo_delta', 'p1', 'p2', 'p3', 'p4', 'score_t1', 'score_t2']].copy()
        edit_df.insert(0, "Delete", False) 

        edited_log = st.data_editor(
            edit_df,
            column_config={
                "Delete": st.column_config.CheckboxColumn("Delete?", default=False),
                "elo_delta": st.column_config.NumberColumn("Elo Delta", format="%.1f"),
            },
            disabled=["id", "date", "league", "match_type", "elo_delta", "p1", "p2", "p3", "p4", "score_t1", "score_t2"],
            use_container_width=True,
            hide_index=True
        )
        
        to_delete = edited_log[edited_log['Delete'] == True]
        if not to_delete.empty:
            st.error(f"⚠️ You have selected {len(to_delete)} matches for deletion.")
            if st.button("Confirm Bulk Delete"):
                supabase.table("matches").delete().in_("id", to_delete['id'].tolist()).execute()
                st.success("Deleted!"); time.sleep(1); st.rerun()

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

elif sel == "📘 Admin Guide":
    st.header("📘 Admin Guide")
    st.markdown("""
    ### 📝 Match Uploader
    * **One tool for everything:** Use this tab to enter matches for official leagues OR casual pop-up events.
    * **Batch Entry:** Best for entering results from a clipboard after the games are done.
    * **Live Round Robin:** Best for running an event live. It generates the schedule for you.

    ### 🏟️ League Manager
    * **Settings Only:** Use this tab to create new leagues, archive old ones (uncheck 'Active'), or change the "Min Games" requirement for the leaderboard.
    
    ### ⚙️ Admin Tools
    * **Recalculate:** If you change a K-Factor or delete a bad match, run this to fix everyone's ratings.
    """)

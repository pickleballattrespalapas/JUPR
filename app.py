import streamlit as st
import pandas as pd
from supabase import create_client, Client
import time
from datetime import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide", page_icon="🌵")

# Clear cache to ensure new columns load
if 'cache_cleared' not in st.session_state:
    st.cache_resource.clear()
    st.session_state.cache_cleared = True

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

            # 3. Matches (Limit 5000)
            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("id", desc=True).limit(5000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
            # 4. Metadata
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
    if len(players) < int(format_type.split('-')[0]): return []
    if format_type == "4-Player": return [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif format_type == "5-Player": return [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5'}]
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

# --- PROCESSOR (With Weeks & End Ratings) ---
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
        week_tag = m.get('week_tag', 'Week 1')
        is_popup = m.get('match_type') == 'PopUp'
        
        # 1. Capture Global Snapshots (Before)
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)
        
        # 2. Calculate
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2, k_factor=DEFAULT_K_FACTOR)
        di1, di2 = 0, 0
        if not is_popup:
            k_val = get_k(league)
            ri1, ri2, ri3, ri4 = get_island_r(p1, league), get_island_r(p2, league), get_island_r(p3, league), get_island_r(p4, league)
            di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2, k_factor=k_val)

        # 3. Apply & Get End Ratings
        win = s1 > s2
        
        # Helper to update and return new rating
        def update_get_new(pid, d_ov, d_isl, won):
            if pid is None: return 0, 0
            
            # Global Update
            if pid not in overall_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                overall_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            overall_updates[pid]['r'] += d_ov
            overall_updates[pid]['mp'] += 1
            if won: overall_updates[pid]['w'] += 1
            else: overall_updates[pid]['l'] += 1
            new_global = overall_updates[pid]['r']
            
            # Island Update
            if not is_popup:
                key = (pid, league)
                if key not in island_updates:
                    curr = get_island_r(pid, league) 
                    island_updates[key] = {'r': curr, 'w': 0, 'l': 0, 'mp': 0}
                island_updates[key]['r'] += d_isl
                island_updates[key]['mp'] += 1
                if won: island_updates[key]['w'] += 1
                else: island_updates[key]['l'] += 1
            
            return new_global, 0 # Return global end for DB storage

        end_r1, _ = update_get_new(p1, do1, di1, win)
        end_r2, _ = update_get_new(p2, do1, di1, win)
        end_r3, _ = update_get_new(p3, do2, di2, not win)
        end_r4, _ = update_get_new(p4, do2, di2, not win)

        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": league,
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, 
            "elo_delta": do1 if s1 > s2 else do2,
            "match_type": m['match_type'],
            "week_tag": week_tag,
            # SNAPSHOTS (Start)
            "t1_p1_r": ro1, "t1_p2_r": ro2, "t2_p1_r": ro3, "t2_p2_r": ro4,
            # SNAPSHOTS (End)
            "t1_p1_r_end": end_r1, "t1_p2_r_end": end_r2, "t2_p1_r_end": end_r3, "t2_p2_r_end": end_r4
        })

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
    
    if not df_meta.empty:
        active_meta = df_meta[df_meta['is_active'] == True]
        available_leagues = ["OVERALL"] + sorted(active_meta['league_name'].unique().tolist())
    else:
        available_leagues = ["OVERALL"]
        
    target_league = st.selectbox("Select View", available_leagues)
    
    # Rules
    min_games_req = 0
    if target_league != "OVERALL" and not df_meta.empty:
        cfg = df_meta[df_meta['league_name'] == target_league]
        if not cfg.empty:
            min_games_req = cfg.iloc[0].get('min_games', 0)

    # Data Prep
    if target_league == "OVERALL":
        display_df = df_players.copy()
    else:
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
    if not display_df.empty and 'rating' in display_df.columns:
        display_df['JUPR'] = (display_df['rating']/400)
        display_df['Win %'] = (display_df['wins'] / display_df['matches_played'].replace(0,1) * 100)
        
        # --- ROBUST GAIN CALCULATION ---
        def calculate_gain(row):
            pid = row['id'] if 'id' in row else row['player_id']
            curr_r = row['rating']
            
            # Default: Compare to starting_rating
            base_start = 1200.0
            if not df_players.empty:
                p_rec = df_players[df_players['id'] == pid]
                if not p_rec.empty: base_start = float(p_rec.iloc[0]['starting_rating'])

            if target_league == "OVERALL" or df_matches.empty:
                return curr_r - base_start

            # Find First Match in League
            relevant = df_matches[
                (df_matches['league'] == target_league) & 
                ((df_matches['t1_p1'] == pid) | (df_matches['t1_p2'] == pid) | (df_matches['t2_p1'] == pid) | (df_matches['t2_p2'] == pid))
            ]
            
            if relevant.empty: return curr_r - base_start
            
            oldest = relevant.iloc[-1] 
            snap = 0
            if pid == oldest['t1_p1']: snap = oldest.get('t1_p1_r', 0)
            elif pid == oldest['t1_p2']: snap = oldest.get('t1_p2_r', 0)
            elif pid == oldest['t2_p1']: snap = oldest.get('t2_p1_r', 0)
            elif pid == oldest['t2_p2']: snap = oldest.get('t2_p2_r', 0)
            
            # Safety: If snapshot is 0/None, fallback to base
            if snap is None or snap == 0: snap = base_start
            
            return curr_r - snap

        display_df['rating_gain'] = display_df.apply(calculate_gain, axis=1)

        # Top Widgets
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

        # Table
        st.markdown("### 📊 Standings")
        final_view = display_df.sort_values('rating', ascending=False).copy()
        final_view['Rank'] = range(1, len(final_view) + 1)
        final_view['Rank'] = final_view['Rank'].apply(lambda r: "🥇" if r==1 else "🥈" if r==2 else "🥉" if r==3 else str(r))
        final_view['Gain'] = (final_view['rating_gain']/400).map('{:+.3f}'.format)
        
        st.dataframe(
            final_view[['Rank', 'name', 'JUPR', 'Gain', 'matches_played', 'wins', 'losses', 'Win %']], 
            use_container_width=True, hide_index=True,
            column_config={
                "Rank": st.column_config.Column("Rank", width="small"),
                "JUPR": st.column_config.NumberColumn("Rating", format="%.3f"),
                "Win %": st.column_config.ProgressColumn("Win %", format="%.1f%%", min_value=0, max_value=100),
            }
        )
    else:
        st.info("No data.")

elif sel == "📝 Match Uploader":
    st.header("📝 Match Uploader")
    
    # 1. Context
    c1, c2, c3 = st.columns(3)
    ctx_type = c1.radio("Context", ["🏆 Official League", "🎉 Pop-Up"], horizontal=True)
    
    selected_league = ""
    is_popup = False
    
    if ctx_type == "🏆 Official League":
        opts = sorted(df_meta[df_meta['is_active']==True]['league_name'].tolist()) if not df_meta.empty else ["Default"]
        selected_league = c2.selectbox("Select League", opts)
        match_type_db = "Live Match"
    else:
        selected_league = c2.text_input("Event Name", "Saturday Social")
        is_popup = True
        match_type_db = "PopUp"

    # 2. WEEK TAG (New!)
    week_tag = c3.selectbox("Week / Session", [f"Week {i}" for i in range(1, 13)] + ["Playoffs", "Finals", "Event"])

    st.divider()
    entry_method = st.radio("Entry Method", ["📋 Manual / Batch", "🏟️ Live Round Robin"], horizontal=True)
    st.write("")

    if entry_method == "📋 Manual / Batch":
        player_list = sorted(df_players['name'].tolist())
        if 'batch_df' not in st.session_state:
            st.session_state.batch_df = pd.DataFrame([{'T1_P1': None, 'T1_P2': None, 'Score_1': 0, 'Score_2': 0, 'T2_P1': None, 'T2_P2': None} for _ in range(5)])

        edited_batch = st.data_editor(st.session_state.batch_df, num_rows="dynamic", use_container_width=True,
            column_config={"T1_P1": st.column_config.SelectboxColumn("T1 P1", options=player_list), "T2_P1": st.column_config.SelectboxColumn("T2 P1", options=player_list), "T1_P2": st.column_config.SelectboxColumn("T1 P2", options=player_list), "T2_P2": st.column_config.SelectboxColumn("T2 P2", options=player_list)})

        if st.button("Submit Batch"):
            valid_batch = []
            for _, row in edited_batch.iterrows():
                if row['T1_P1'] and row['T2_P1'] and (row['Score_1'] + row['Score_2'] > 0):
                    valid_batch.append({
                        't1_p1': row['T1_P1'], 't1_p2': row['T1_P2'], 't2_p1': row['T2_P1'], 't2_p2': row['T2_P2'], 
                        's1': int(row['Score_1']), 's2': int(row['Score_2']), 
                        'date': str(datetime.now()), 'league': selected_league, 'match_type': match_type_db, 
                        'week_tag': week_tag # Saved!
                    })
            if valid_batch:
                process_matches(valid_batch, name_to_id, df_players, df_leagues, df_meta)
                st.success("✅ Processed!"); time.sleep(1); st.rerun()

    else:
        # Live RR Logic (Simplified for brevity, similar to batch but with schedule generator)
        if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
        st.session_state.lc_courts = st.number_input("Courts", 1, 10, st.session_state.lc_courts)
        with st.form("setup_lc"):
            c_data = []
            for i in range(st.session_state.lc_courts):
                c1, c2 = st.columns([1,3])
                t = c1.selectbox(f"F{i}", ["4-Player","5-Player"], key=f"t_{i}")
                n = c2.text_area(f"P{i}", height=70, key=f"n_{i}")
                c_data.append({'type':t, 'names':n})
            if st.form_submit_button("Generate"):
                st.session_state.lc_schedule = []
                st.session_state.active_lg = selected_league
                st.session_state.active_wk = week_tag
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl)})
                st.rerun()
        
        if 'lc_schedule' in st.session_state:
            with st.form("scores_lc"):
                all_res = []
                for c in st.session_state.lc_schedule:
                    st.markdown(f"**Court {c['c']}**")
                    for i, m in enumerate(c['m']):
                        c1,c2,c3,c4 = st.columns([3,1,1,3])
                        c1.text(f"{m['t1'][0]}/{m['t1'][1]}"); s1=c2.number_input("S1",0,key=f"s1_{c['c']}_{i}"); s2=c3.number_input("S2",0,key=f"s2_{c['c']}_{i}"); c4.text(f"{m['t2'][0]}/{m['t2'][1]}")
                        all_res.append({'t1_p1':m['t1'][0],'t1_p2':m['t1'][1],'t2_p1':m['t2'][0],'t2_p2':m['t2'][1],'s1':s1,'s2':s2,'date':str(datetime.now()),'league':st.session_state.active_lg,'match_type':match_type_db,'week_tag':st.session_state.active_wk})
                if st.form_submit_button("Submit"):
                     process_matches([x for x in all_res if x['s1']>0 or x['s2']>0], name_to_id, df_players, df_leagues, df_meta)
                     st.success("✅ Done!"); del st.session_state.lc_schedule; time.sleep(1); st.rerun()

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    target_reset = st.selectbox("Recalculate League", ["ALL (Full System Reset)"] + sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else ["ALL"])
    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching..."):
            all_players = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute().data
            all_matches = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("date", desc=False).execute().data
            p_map = {p['id']: {'r': float(p['starting_rating']), 'w': 0, 'l': 0, 'mp': 0} for p in all_players}
            island_map = {}
            matches_to_update = []
            
            for m in all_matches:
                if target_reset != "ALL (Full System Reset)" and m['league'] != target_reset: continue
                p1,p2,p3,p4 = m['t1_p1'],m['t1_p2'],m['t2_p1'],m['t2_p2']; s1,s2=m['score_t1'],m['score_t2']
                
                def gr(pid): return p_map[pid]['r'] if pid else 1200.0
                sr1,sr2,sr3,sr4 = gr(p1),gr(p2),gr(p3),gr(p4) # Start Snapshot
                
                do1,do2 = calculate_hybrid_elo((sr1+sr2)/2,(sr3+sr4)/2,s1,s2)
                
                win=s1>s2
                for pid,d,w in [(p1,do1,win),(p2,do1,win),(p3,do2,not win),(p4,do2,not win)]:
                    if pid: p_map[pid]['r']+=d; p_map[pid]['mp']+=1; p_map[pid]['w' if w else 'l']+=1
                
                er1,er2,er3,er4 = gr(p1),gr(p2),gr(p3),gr(p4) # End Snapshot

                # Island Logic (Simulated)
                if m.get('match_type') != 'PopUp':
                    def gir(pid, lg): 
                        if (pid,lg) not in island_map: island_map[(pid,lg)]={'r':p_map[pid]['r'],'w':0,'l':0,'mp':0}
                        return island_map[(pid,lg)]['r']
                    ir1,ir2,ir3,ir4 = gir(p1,m['league']),gir(p2,m['league']),gir(p3,m['league']),gir(p4,m['league'])
                    di1,di2 = calculate_hybrid_elo((ir1+ir2)/2,(ir3+ir4)/2,s1,s2)
                    for pid,d,w in [(p1,di1,win),(p2,di1,win),(p3,di2,not win),(p4,di2,not win)]:
                        if pid: k=(pid,m['league']); island_map[k]['r']+=d; island_map[k]['mp']+=1; island_map[k]['w' if w else 'l']+=1

                matches_to_update.append({'id':m['id'], 'elo_delta':do1 if s1>s2 else do2, 't1_p1_r':sr1,'t1_p2_r':sr2,'t2_p1_r':sr3,'t2_p2_r':sr4, 't1_p1_r_end':er1,'t1_p2_r_end':er2,'t2_p1_r_end':er3,'t2_p2_r_end':er4})

            # Save
            if target_reset == "ALL (Full System Reset)":
                 for pid,s in p_map.items(): supabase.table("players").update({"rating":s['r'],"wins":s['w'],"losses":s['l'],"matches_played":s['mp']}).eq("id",pid).execute()
            
            if target_reset != "ALL (Full System Reset)": supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", target_reset).execute()
            else: supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()
            
            new_is = []
            for (pid,lg),s in island_map.items():
                if target_reset == "ALL (Full System Reset)" or lg==target_reset: new_is.append({"club_id":CLUB_ID,"player_id":pid,"league_name":lg,"rating":s['r'],"wins":s['w'],"losses":s['l'],"matches_played":s['mp']})
            if new_is: 
                for i in range(0,len(new_is),1000): supabase.table("league_ratings").insert(new_is[i:i+1000]).execute()
            
            bar = st.progress(0)
            for i,u in enumerate(matches_to_update): supabase.table("matches").update(u).eq("id",u['id']).execute(); bar.progress((i+1)/len(matches_to_update))
            st.success("Done!"); time.sleep(1); st.rerun()

# Keep other tabs (Search, Players, etc) as they were or simplified.
elif sel == "🔍 Player Search":
    st.header("🔍 Player History")
    p = st.selectbox("Search", [""] + sorted(df_players['name']))
    if p and not df_matches.empty:
        mask = (df_matches['p1'] == p) | (df_matches['p2'] == p) | (df_matches['p3'] == p) | (df_matches['p4'] == p)
        h = df_matches[mask].copy()
        if not h.empty:
            def get_snap(r):
                if p==r['p1']: return r.get('t1_p1_r',0)
                if p==r['p2']: return r.get('t1_p2_r',0)
                if p==r['p3']: return r.get('t2_p1_r',0)
                if p==r['p4']: return r.get('t2_p2_r',0)
                return 0
            h['Rating Before'] = h.apply(get_snap, axis=1).apply(lambda x: f"{x/400:.3f}" if x else "-")
            h['Result'] = h.apply(lambda x: "✅ Win" if (p in [x['p1'], x['p2']] and x['score_t1']>x['score_t2']) or (p in [x['p3'], x['p4']] and x['score_t2']>x['score_t1']) else "❌ Loss", axis=1)
            st.dataframe(h[['date','league','week_tag','Result','Rating Before','score_t1','score_t2']], use_container_width=True, hide_index=True)

elif sel == "🏟️ League Manager":
    st.header("🏟️ League Manager (Settings)")
    if not df_meta.empty:
        editor = st.data_editor(df_meta[['id','league_name','is_active','min_games','description','k_factor']], disabled=['id','league_name'], hide_index=True, use_container_width=True)
        if st.button("Save"):
            for _,r in editor.iterrows(): supabase.table("leagues_metadata").update({"is_active":r['is_active'],"min_games":r['min_games'],"k_factor":r['k_factor']}).eq("id",r['id']).execute()
            st.rerun()
    with st.form("new"):
        if st.form_submit_button("Create New League"):
            n=st.text_input("Name"); supabase.table("leagues_metadata").insert({"club_id":CLUB_ID,"league_name":n,"is_active":True}).execute(); st.rerun()

elif sel == "👥 Players":
    st.header("Players")
    with st.form("add"):
        n=st.text_input("Name"); r=st.number_input("Rating",1.0,7.0,3.5)
        if st.form_submit_button("Add"): safe_add_player(n,r); st.rerun()
    st.dataframe(df_players)

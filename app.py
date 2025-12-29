import streamlit as st
import pandas as pd
from supabase import create_client, Client
import time
from datetime import datetime
import re
import altair as alt

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
    .court-header { background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-weight: bold; margin-top: 20px; font-size: 1.2rem; }
    .move-up { color: #28a745; font-weight: bold; font-size: 1.1rem; }
    .move-down { color: #dc3545; font-weight: bold; font-size: 1.1rem; }
    .move-stay { color: #6c757d; font-weight: bold; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
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
            p_response = supabase.table("players").select("*").eq("club_id", CLUB_ID).execute()
            df_players = pd.DataFrame(p_response.data)
            # Add .eq("active", True) to your fetch queries
            response = supabase.table("players").select("*").eq("active", True).execute()
            
            l_response = supabase.table("league_ratings").select("*").eq("club_id", CLUB_ID).execute()
            df_leagues = pd.DataFrame(l_response.data)

            m_response = supabase.table("matches").select("*").eq("club_id", CLUB_ID).order("id", desc=True).limit(5000).execute()
            df_matches = pd.DataFrame(m_response.data)
            
            meta_response = supabase.table("leagues_metadata").select("*").eq("club_id", CLUB_ID).execute()
            df_meta = pd.DataFrame(meta_response.data)
            
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
            if attempt < max_retries - 1: time.sleep(1); continue
            else: st.error(f"⚠️ Network Error: {e}"); return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

# --- HELPERS ---
def get_match_schedule(format_type, players, custom_text=None):
    p = players
    if custom_text and len(custom_text.strip()) > 5:
        matches = []
        lines = custom_text.strip().split('\n')
        r_num = 1
        for line in lines:
            nums = [int(x) for x in re.findall(r'\d+', line)]
            if len(nums) >= 4:
                idx = [n-1 for n in nums[:4]]
                if all(0 <= i < len(p) for i in idx):
                    matches.append({'t1': [p[idx[0]], p[idx[1]]], 't2': [p[idx[2]], p[idx[3]]], 'desc': f"Game {r_num}"})
                    r_num += 1
        if matches: return matches

    if len(p) < int(format_type.split('-')[0]): return []
    if format_type == "4-Player": 
        return [{'t1': [p[1], p[0]], 't2': [p[2], p[3]], 'desc': 'Rnd 1'}, {'t1': [p[3], p[1]], 't2': [p[0], p[2]], 'desc': 'Rnd 2'}, {'t1': [p[3], p[0]], 't2': [p[1], p[2]], 'desc': 'Rnd 3'}]
    elif format_type == "5-Player": 
        return [{'t1': [p[0], p[1]], 't2': [p[2], p[3]], 'desc': 'Rnd 1'}, {'t1': [p[1], p[3]], 't2': [p[2], p[4]], 'desc': 'Rnd 2'}, {'t1': [p[0], p[4]], 't2': [p[1], p[2]], 'desc': 'Rnd 3'}, {'t1': [p[0], p[2]], 't2': [p[3], p[4]], 'desc': 'Rnd 4'}, {'t1': [p[0], p[3]], 't2': [p[1], p[4]], 'desc': 'Rnd 5'}]
    elif format_type == "6-Player": 
        return [{'t1':[p[0],p[1]],'t2':[p[2],p[4]],'desc':'R1'}, {'t1':[p[2],p[5]],'t2':[p[0],p[4]],'desc':'R2'}, {'t1':[p[1],p[3]],'t2':[p[4],p[5]],'desc':'R3'}, {'t1':[p[0],p[5]],'t2':[p[1],p[2]],'desc':'R4'}, {'t1':[p[0],p[3]],'t2':[p[1],p[4]],'desc':'R5'}]
    elif format_type == "8-Player":
        return [{'t1': [p[0], p[5]], 't2': [p[1], p[4]], 'desc': 'Rnd 1 (Ct 1)'}, {'t1': [p[2], p[7]], 't2': [p[3], p[6]], 'desc': 'Rnd 1 (Ct 2)'}, {'t1': [p[1], p[2]], 't2': [p[4], p[7]], 'desc': 'Rnd 2 (Ct 1)'}, {'t1': [p[0], p[3]], 't2': [p[5], p[6]], 'desc': 'Rnd 2 (Ct 2)'}, {'t1': [p[0], p[7]], 't2': [p[2], p[5]], 'desc': 'Rnd 3 (Ct 1)'}, {'t1': [p[1], p[6]], 't2': [p[3], p[4]], 'desc': 'Rnd 3 (Ct 2)'}, {'t1': [p[0], p[1]], 't2': [p[2], p[3]], 'desc': 'Rnd 4 (Ct 1)'}, {'t1': [p[4], p[5]], 't2': [p[6], p[7]], 'desc': 'Rnd 4 (Ct 2)'}, {'t1': [p[0], p[6]], 't2': [p[1], p[7]], 'desc': 'Rnd 5 (Ct 1)'}, {'t1': [p[2], p[4]], 't2': [p[3], p[5]], 'desc': 'Rnd 5 (Ct 2)'}, {'t1': [p[1], p[5]], 't2': [p[2], p[6]], 'desc': 'Rnd 6 (Ct 1)'}, {'t1': [p[0], p[4]], 't2': [p[3], p[7]], 'desc': 'Rnd 6 (Ct 2)'}, {'t1': [p[1], p[3]], 't2': [p[5], p[7]], 'desc': 'Rnd 7 (Ct 1)'}, {'t1': [p[0], p[2]], 't2': [p[4], p[6]], 'desc': 'Rnd 7 (Ct 2)'}]
    elif format_type == "12-Player":
        return [{'t1': [p[2], p[5]], 't2': [p[3], p[10]], 'desc': 'Rnd 1 (Ct 1)'}, {'t1': [p[4], p[6]], 't2': [p[8], p[9]], 'desc': 'Rnd 1 (Ct 2)'}, {'t1': [p[11], p[0]], 't2': [p[1], p[7]], 'desc': 'Rnd 1 (Ct 3)'}, {'t1': [p[5], p[8]], 't2': [p[6], p[2]], 'desc': 'Rnd 2 (Ct 1)'}, {'t1': [p[7], p[9]], 't2': [p[0], p[1]], 'desc': 'Rnd 2 (Ct 2)'}, {'t1': [p[11], p[3]], 't2': [p[4], p[10]], 'desc': 'Rnd 2 (Ct 3)'}, {'t1': [p[10], p[1]], 't2': [p[3], p[4]], 'desc': 'Rnd 3 (Ct 1)'}, {'t1': [p[11], p[6]], 't2': [p[7], p[2]], 'desc': 'Rnd 3 (Ct 2)'}, {'t1': [p[8], p[0]], 't2': [p[9], p[5]], 'desc': 'Rnd 3 (Ct 3)'}, {'t1': [p[11], p[9]], 't2': [p[10], p[5]], 'desc': 'Rnd 4 (Ct 1)'}, {'t1': [p[0], p[3]], 't2': [p[1], p[8]], 'desc': 'Rnd 4 (Ct 2)'}, {'t1': [p[2], p[4]], 't2': [p[6], p[7]], 'desc': 'Rnd 4 (Ct 3)'}, {'t1': [p[3], p[6]], 't2': [p[4], p[0]], 'desc': 'Rnd 5 (Ct 1)'}, {'t1': [p[5], p[7]], 't2': [p[9], p[10]], 'desc': 'Rnd 5 (Ct 2)'}, {'t1': [p[11], p[1]], 't2': [p[2], p[8]], 'desc': 'Rnd 5 (Ct 3)'}, {'t1': [p[8], p[10]], 't2': [p[1], p[2]], 'desc': 'Rnd 6 (Ct 1)'}, {'t1': [p[11], p[4]], 't2': [p[5], p[0]], 'desc': 'Rnd 6 (Ct 2)'}, {'t1': [p[6], p[9]], 't2': [p[7], p[3]], 'desc': 'Rnd 6 (Ct 3)'}, {'t1': [p[11], p[7]], 't2': [p[8], p[3]], 'desc': 'Rnd 7 (Ct 1)'}, {'t1': [p[9], p[1]], 't2': [p[10], p[6]], 'desc': 'Rnd 7 (Ct 2)'}, {'t1': [p[0], p[2]], 't2': [p[4], p[5]], 'desc': 'Rnd 7 (Ct 3)'}, {'t1': [p[1], p[4]], 't2': [p[2], p[9]], 'desc': 'Rnd 8 (Ct 1)'}, {'t1': [p[3], p[5]], 't2': [p[7], p[8]], 'desc': 'Rnd 8 (Ct 2)'}, {'t1': [p[11], p[10]], 't2': [p[0], p[6]], 'desc': 'Rnd 8 (Ct 3)'}, {'t1': [p[6], p[8]], 't2': [p[10], p[0]], 'desc': 'Rnd 9 (Ct 1)'}, {'t1': [p[4], p[7]], 't2': [p[5], p[1]], 'desc': 'Rnd 9 (Ct 2)'}, {'t1': [p[11], p[2]], 't2': [p[3], p[9]], 'desc': 'Rnd 9 (Ct 3)'}, {'t1': [p[11], p[5]], 't2': [p[6], p[1]], 'desc': 'Rnd 10 (Ct 1)'}, {'t1': [p[9], p[0]], 't2': [p[2], p[3]], 'desc': 'Rnd 10 (Ct 2)'}, {'t1': [p[7], p[10]], 't2': [p[8], p[4]], 'desc': 'Rnd 10 (Ct 3)'}, {'t1': [p[10], p[2]], 't2': [p[0], p[7]], 'desc': 'Rnd 11 (Ct 1)'}, {'t1': [p[11], p[8]], 't2': [p[9], p[4]], 'desc': 'Rnd 11 (Ct 2)'}, {'t1': [p[1], p[3]], 't2': [p[5], p[6]], 'desc': 'Rnd 11 (Ct 3)'}]
    return []

def safe_add_player(name, rating):
    try:
        supabase.table("players").insert({
            "club_id": CLUB_ID, "name": name, 
            "rating": rating * 400, "starting_rating": rating * 400
        }).execute()
        return True, ""
    except Exception as e: return False, str(e)

# --- PROCESSOR ---
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
        is_popup = m.get('match_type') == 'PopUp' or m.get('is_popup', False)
        
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)
        
        do1, do2 = calculate_hybrid_elo((ro1+ro2)/2, (ro3+ro4)/2, s1, s2, k_factor=DEFAULT_K_FACTOR)
        di1, di2 = 0, 0
        if not is_popup:
            k_val = get_k(league)
            ri1, ri2, ri3, ri4 = get_island_r(p1, league), get_island_r(p2, league), get_island_r(p3, league), get_island_r(p4, league)
            di1, di2 = calculate_hybrid_elo((ri1+ri2)/2, (ri3+ri4)/2, s1, s2, k_factor=k_val)

        win = s1 > s2
        
        def update_get_new(pid, d_ov, d_isl, won):
            if pid is None: return 0
            if pid not in overall_updates:
                row = df_p[df_p['id'] == pid].iloc[0]
                overall_updates[pid] = {'r': float(row['rating']), 'w': int(row['wins']), 'l': int(row['losses']), 'mp': int(row['matches_played'])}
            overall_updates[pid]['r'] += d_ov
            overall_updates[pid]['mp'] += 1
            if won: overall_updates[pid]['w'] += 1
            else: overall_updates[pid]['l'] += 1
            new_global = overall_updates[pid]['r']
            
            if not is_popup:
                key = (pid, league)
                if key not in island_updates:
                    curr = get_island_r(pid, league) 
                    island_updates[key] = {'r': curr, 'w': 0, 'l': 0, 'mp': 0}
                island_updates[key]['r'] += d_isl
                island_updates[key]['mp'] += 1
                if won: island_updates[key]['w'] += 1
                else: island_updates[key]['l'] += 1
            return new_global

        end_r1 = update_get_new(p1, do1, di1, win)
        end_r2 = update_get_new(p2, do1, di1, win)
        end_r3 = update_get_new(p3, do2, di2, not win)
        end_r4 = update_get_new(p4, do2, di2, not win)

        db_matches.append({
            "club_id": CLUB_ID, "date": m['date'], "league": league,
            "t1_p1": p1, "t1_p2": p2, "t2_p1": p3, "t2_p2": p4,
            "score_t1": s1, "score_t2": s2, "elo_delta": do1 if s1 > s2 else do2,
            "match_type": m['match_type'], "week_tag": week_tag,
            "t1_p1_r": ro1, "t1_p2_r": ro2, "t2_p1_r": ro3, "t2_p2_r": ro4,
            "t1_p1_r_end": end_r1, "t1_p2_r_end": end_r2, "t2_p1_r_end": end_r3, "t2_p2_r_end": end_r4
        })

    if db_matches: supabase.table("matches").insert(db_matches).execute()
    for pid, stats in overall_updates.items():
        supabase.table("players").update({"rating": stats['r'], "wins": stats['w'], "losses": stats['l'], "matches_played": stats['mp']}).eq("id", pid).execute()
    if island_updates:
        island_data = []
        for (pid, league), stats in island_updates.items():
            island_data.append({"player_id": pid, "club_id": CLUB_ID, "league_name": league, "rating": stats['r'], "matches_played": stats['mp'], "wins": stats['w'], "losses": stats['l']})
        for row in island_data:
            existing = supabase.table("league_ratings").select("*").eq("player_id", row['player_id']).eq("league_name", row['league_name']).execute()
            if existing.data:
                cur = existing.data[0]
                row['wins'] += cur['wins']; row['losses'] += cur['losses']; row['matches_played'] += cur['matches_played']
                supabase.table("league_ratings").update(row).eq("id", cur['id']).execute()
            else: supabase.table("league_ratings").insert(row).execute()

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
    nav += ["──────────", "🏟️ League Manager", "📝 Match Uploader", "👥 Player Editor", "📝 Match Log", "⚙️ Admin Tools", "📘 Admin Guide"]
sel = st.sidebar.radio("Go to:", nav, key="main_nav")

# --- UI LOGIC ---

if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")
    if not df_meta.empty:
        active_meta = df_meta[df_meta['is_active'] == True]
        available_leagues = ["OVERALL"] + sorted(active_meta['league_name'].unique().tolist())
    else: available_leagues = ["OVERALL"]
    target_league = st.selectbox("Select View", available_leagues)
    min_games_req = 0
    if target_league != "OVERALL" and not df_meta.empty:
        cfg = df_meta[df_meta['league_name'] == target_league]
        if not cfg.empty: min_games_req = cfg.iloc[0].get('min_games', 0)

    if target_league == "OVERALL": display_df = df_players.copy()
    else:
        if df_leagues.empty: display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues['league_name'] == target_league].copy()
            display_df['name'] = display_df['player_id'].map(id_to_name)
    
    if not display_df.empty and 'rating' in display_df.columns:
        display_df['JUPR'] = (display_df['rating']/400)
        display_df['Win %'] = (display_df['wins'] / display_df['matches_played'].replace(0,1) * 100)
        
        def calculate_gain(row):
            pid = row['id'] if 'id' in row else row['player_id']
            curr_r = row['rating']
            base_start = 1200.0
            if not df_players.empty:
                p_rec = df_players[df_players['id'] == pid]
                if not p_rec.empty: base_start = float(p_rec.iloc[0]['starting_rating'])
            if target_league == "OVERALL" or df_matches.empty: return curr_r - base_start
            relevant = df_matches[(df_matches['league'] == target_league) & ((df_matches['t1_p1'] == pid) | (df_matches['t1_p2'] == pid) | (df_matches['t2_p1'] == pid) | (df_matches['t2_p2'] == pid))]
            if relevant.empty: return curr_r - base_start
            oldest = relevant.iloc[-1] 
            snap = 0
            if pid == oldest['t1_p1']: snap = oldest.get('t1_p1_r', 0)
            elif pid == oldest['t1_p2']: snap = oldest.get('t1_p2_r', 0)
            elif pid == oldest['t2_p1']: snap = oldest.get('t2_p1_r', 0)
            elif pid == oldest['t2_p2']: snap = oldest.get('t2_p2_r', 0)
            if snap is None or snap == 0: snap = base_start
            return curr_r - snap

        display_df['rating_gain'] = display_df.apply(calculate_gain, axis=1)

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
                    for _, r in top.iterrows(): st.markdown(f"**{'+' if r['rating_gain']>0 else ''}{r['rating_gain']/400:.3f}** - {r['name']}")
                with c3:
                    st.markdown("**🎯 Best Win %**")
                    top = qualified_df.sort_values('Win %', ascending=False).head(5)
                    for _, r in top.iterrows(): st.markdown(f"**{r['Win %']:.1f}%** - {r['name']}")
                with c4:
                    st.markdown("**🚜 Most Wins**")
                    top = qualified_df.sort_values('wins', ascending=False).head(5)
                    for _, r in top.iterrows(): st.markdown(f"**{r['wins']} Wins** - {r['name']}")
                st.divider()

        st.markdown("### 📊 Standings")
        final_view = display_df.sort_values('rating', ascending=False).copy()
        final_view['Rank'] = range(1, len(final_view) + 1)
        final_view['Rank'] = final_view['Rank'].apply(lambda r: "🥇" if r==1 else "🥈" if r==2 else "🥉" if r==3 else str(r))
        final_view['Gain'] = (final_view['rating_gain']/400).map('{:+.3f}'.format)
        st.dataframe(final_view[['Rank', 'name', 'JUPR', 'Gain', 'matches_played', 'wins', 'losses', 'Win %']], use_container_width=True, hide_index=True)
    else: st.info("No data.")

elif sel == "🔍 Player Search":
        st.header("🕵️ Player Search & Audit")

        # 1. FETCH ACTIVE PLAYERS
        players_response = supabase.table("players").select("id, name, rating").eq("active", True).execute()
        players_df = pd.DataFrame(players_response.data)

        if players_df.empty:
            st.warning("No active players found.")
        else:
            # Dropdown
            player_names = sorted(players_df['name'].tolist())
            selected_name = st.selectbox("Select a Player:", player_names)

            selected_player = players_df[players_df['name'] == selected_name].iloc[0]
            p_id = int(selected_player['id'])
            
            # --- TOP METRICS ---
            col1, col2 = st.columns(2)
            col1.metric("Player Name", selected_player['name'])
            col2.metric("Current JUPR", selected_player['rating'])
            
            # 2. FETCH MATCH HISTORY
            response = supabase.table("matches").select("*").or_(f"player1_id.eq.{p_id},player2_id.eq.{p_id},player3_id.eq.{p_id},player4_id.eq.{p_id}").order("match_date", desc=True).execute()
            matches_data = response.data

            if not matches_data:
                st.info("This player has no recorded matches yet.")
            else:
                # --- NEW LOGIC: PARSE SCORE TO FIND WINNER ---
                processed_matches = []
                
                for match in matches_data:
                    # A. Figure out which team the selected player was on
                    if match['player1_id'] == p_id or match['player2_id'] == p_id:
                        my_team = 1
                    else:
                        my_team = 2
                    
                    # B. Parse the Score (e.g., "11-9") to find the winner
                    score_text = str(match['score'])
                    winner_team = 0 # Default to unknown
                    
                    try:
                        # Assumes format "11-9"
                        parts = score_text.split('-')
                        score1 = int(parts[0])
                        score2 = int(parts[1])
                        
                        if score1 > score2:
                            winner_team = 1
                        else:
                            winner_team = 2
                    except:
                        # If score format is weird (e.g. "Win", "Forfeit"), we can't guess.
                        pass

                    # C. Determine +/- Sign
                    raw_change = match['elo_change']
                    
                    if winner_team == 0:
                        # Fallback: if we can't read the score, assume positive to be safe
                        final_change = raw_change 
                    elif winner_team == my_team:
                        final_change = abs(raw_change) # Ensure it's positive
                    else:
                        final_change = -1 * abs(raw_change) # Ensure it's negative

                    processed_matches.append({
                        'Date': match['match_date'],
                        'Score': match['score'],
                        'JUPR Change': final_change
                    })

                # Create DataFrame
                display_df = pd.DataFrame(processed_matches)
                display_df['Date'] = pd.to_datetime(display_df['Date'])

                # 4. THE "AUDITOR" GRAPH
                st.subheader("JUPR Movement History")
                
                chart = alt.Chart(display_df.head(20)).mark_bar().encode(
                    x='Date',
                    y='JUPR Change',
                    color=alt.condition(
                        alt.datum['JUPR Change'] > 0,
                        alt.value("green"),
                        alt.value("red")
                    ),
                    tooltip=['Date', 'Score', 'JUPR Change']
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)

                # 5. DETAILED TABLE
                st.subheader("Match Log")
                st.dataframe(
                    display_df[['Date', 'Score', 'JUPR Change']], 
                    use_container_width=True,
                    hide_index=True
                )

elif sel == "🏟️ League Manager":
    st.header("🏟️ League Manager")
    tabs = st.tabs(["🏃‍♂️ Run Live Event (Ladder)", "⚙️ Settings"])
    
    # --- TAB 1: LIVE LADDER SYSTEM ---
    with tabs[0]:
        st.subheader("Ladder Management")
        
        if 'ladder_state' not in st.session_state: st.session_state.ladder_state = 'SETUP'
        if 'ladder_roster' not in st.session_state: st.session_state.ladder_roster = []
        if 'ladder_total_rounds' not in st.session_state: st.session_state.ladder_total_rounds = 5
        
        # 1. SETUP PHASE
        if st.session_state.ladder_state == 'SETUP':
            st.markdown("#### Step 1: Select League & Roster")
            opts = sorted(df_meta[df_meta['is_active']==True]['league_name'].tolist()) if not df_meta.empty else ["Default"]
            lg_select = st.selectbox("Select League", opts, key="ladder_lg")
            week_select = st.selectbox("Week", [f"Week {i}" for i in range(1, 13)] + ["Playoffs"], key="ladder_wk")
            num_rounds = st.number_input("Total Rounds to Play", 1, 20, 5)
            
            raw = st.text_area("Paste Player List (one per line)", height=150)
            
            if st.button("Analyze & Seed"):
                # SAVE PERMANENT STATE
                st.session_state.saved_ladder_lg = lg_select
                st.session_state.saved_ladder_wk = week_select
                st.session_state.ladder_total_rounds = num_rounds
                
                parsed = [x.strip() for x in raw.replace('\n',',').split(',') if x.strip()]
                roster_data = []
                new_ps = []
                
                for n in parsed:
                    if n in name_to_id:
                        pid = name_to_id[n]
                        r = 1200.0
                        if not df_leagues.empty:
                            row = df_leagues[(df_leagues['player_id'] == pid) & (df_leagues['league_name'] == lg_select)]
                            if not row.empty: r = float(row.iloc[0]['rating'])
                            else:
                                row_g = df_players[df_players['id'] == pid]
                                if not row_g.empty: r = float(row_g.iloc[0]['rating'])
                        else:
                            row_g = df_players[df_players['id'] == pid]
                            if not row_g.empty: r = float(row_g.iloc[0]['rating'])
                        
                        roster_data.append({'name': n, 'rating': r, 'id': pid, 'status': 'Found'})
                    else:
                        new_ps.append(n)
                
                st.session_state.ladder_temp_roster = roster_data
                st.session_state.ladder_temp_new = new_ps
                st.session_state.ladder_state = 'REVIEW_ROSTER'
                st.rerun()

        # 2. REVIEW / NEW PLAYERS
        if st.session_state.ladder_state == 'REVIEW_ROSTER':
            st.markdown("#### Step 2: Confirm Roster")
            if st.session_state.ladder_temp_new:
                st.warning(f"Found {len(st.session_state.ladder_temp_new)} new players.")
                df_new = pd.DataFrame({'Name': st.session_state.ladder_temp_new, 'Rating': [3.5]*len(st.session_state.ladder_temp_new)})
                edited_new = st.data_editor(df_new, column_config={"Rating": st.column_config.NumberColumn(min_value=1.0, max_value=7.0, step=0.1)}, hide_index=True)
                
                if st.button("Save New Players & Continue"):
                    for _, r in edited_new.iterrows():
                        safe_add_player(r['Name'], r['Rating'])
                        st.session_state.ladder_temp_roster.append({'name': r['Name'], 'rating': r['Rating']*400, 'id': None, 'status': 'New'})
                    
                    st.session_state.ladder_roster = sorted(st.session_state.ladder_temp_roster, key=lambda x: x['rating'], reverse=True)
                    st.session_state.ladder_state = 'CONFIG_COURTS'
                    st.rerun()
            else:
                st.success("All players found.")
                if st.button("Proceed to Court Setup"):
                    st.session_state.ladder_roster = sorted(st.session_state.ladder_temp_roster, key=lambda x: x['rating'], reverse=True)
                    st.session_state.ladder_state = 'CONFIG_COURTS'
                    st.rerun()

        # 3. CONFIG COURTS
        if st.session_state.ladder_state == 'CONFIG_COURTS':
            st.markdown("#### Step 3: Configure Courts")
            total_p = len(st.session_state.ladder_roster)
            st.info(f"Total Players: {total_p}")
            
            num_courts = st.number_input("Number of Courts", 1, 10, 3)
            court_sizes = []
            cols = st.columns(num_courts)
            for i in range(num_courts):
                with cols[i]:
                    s = st.number_input(f"Ct {i+1} Size", 4, 12, 4, key=f"cs_{i}")
                    court_sizes.append(s)
            
            if sum(court_sizes) != total_p:
                st.error(f"Court sizes sum to {sum(court_sizes)}, but you have {total_p} players.")
            else:
                if st.button("Preview Assignments"):
                    current_idx = 0
                    final_assignments = []
                    for c_idx, size in enumerate(court_sizes):
                        group = st.session_state.ladder_roster[current_idx : current_idx + size]
                        for p in group:
                            p['court'] = c_idx + 1
                            final_assignments.append(p)
                        current_idx += size
                    
                    st.session_state.ladder_live_roster = pd.DataFrame(final_assignments)
                    st.session_state.ladder_court_sizes = court_sizes
                    st.session_state.ladder_state = 'CONFIRM_START' 
                    st.rerun()

        # 3.5. CONFIRM START
        if st.session_state.ladder_state == 'CONFIRM_START':
            st.markdown("#### Step 4: Confirm Starting Positions")
            st.markdown("Verify the court assignments. Ratings shown in **JUPR**.")
            
            # Display Copy with JUPR Conversion
            display_roster = st.session_state.ladder_live_roster.copy()
            display_roster['JUPR Rating'] = display_roster['rating'] / 400
            
            edited_roster = st.data_editor(
                display_roster[['name', 'JUPR Rating', 'court']],
                column_config={
                    "court": st.column_config.NumberColumn("Court", min_value=1, max_value=10, step=1),
                    "JUPR Rating": st.column_config.NumberColumn("JUPR Rating", format="%.3f", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("✅ Start Event (Round 1)"):
                final_roster = edited_roster.copy()
                final_roster['rating'] = final_roster['JUPR Rating'] * 400 # Convert back just in case, though we rely on name matching mostly
                final_roster = final_roster.sort_values('court')
                
                # Recalc sizes
                new_sizes = final_roster['court'].value_counts().sort_index().tolist()
                st.session_state.ladder_court_sizes = new_sizes
                st.session_state.ladder_live_roster = final_roster
                
                st.session_state.ladder_round_num = 1
                st.session_state.ladder_state = 'PLAY_ROUND'
                st.rerun()

        # 4. PLAY ROUND
        if st.session_state.ladder_state == 'PLAY_ROUND':
            current_r = st.session_state.ladder_round_num
            total_r = st.session_state.ladder_total_rounds
            
            st.markdown(f"### 🎾 Round {current_r} / {total_r}")
            
            if 'current_schedule' not in st.session_state:
                schedule = []
                for c_idx in range(len(st.session_state.ladder_court_sizes)):
                    c_num = c_idx + 1
                    players = st.session_state.ladder_live_roster[st.session_state.ladder_live_roster['court'] == c_num]['name'].tolist()
                    fmt = f"{len(players)}-Player"
                    matches = get_match_schedule(fmt, players)
                    schedule.append({'c': c_num, 'matches': matches})
                st.session_state.current_schedule = schedule

            all_results = []
            with st.form("round_score_form"):
                for c_data in st.session_state.current_schedule:
                    st.markdown(f"<div class='court-header'>Court {c_data['c']}</div>", unsafe_allow_html=True)
                    for m_idx, m in enumerate(c_data['matches']):
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                        c1.text(f"{m['t1'][0]} & {m['t1'][1]}")
                        s1 = c2.number_input("S1", 0, key=f"r{current_r}_c{c_data['c']}_m{m_idx}_1")
                        s2 = c3.number_input("S2", 0, key=f"r{current_r}_c{c_data['c']}_m{m_idx}_2")
                        c4.text(f"{m['t2'][0]} & {m['t2'][1]}")
                        all_results.append({'court': c_data['c'], 't1_p1': m['t1'][0], 't1_p2': m['t1'][1], 't2_p1': m['t2'][0], 't2_p2': m['t2'][1], 's1': s1, 's2': s2})
                
                if st.form_submit_button("Submit Round & Calculate Movement"):
                    valid_matches = []
                    for r in all_results:
                        if r['s1'] > 0 or r['s2'] > 0:
                            valid_matches.append({
                                't1_p1': r['t1_p1'], 't1_p2': r['t1_p2'], 't2_p1': r['t2_p1'], 't2_p2': r['t2_p2'],
                                's1': r['s1'], 's2': r['s2'], 'date': str(datetime.now()),
                                'league': st.session_state.saved_ladder_lg, 
                                'match_type': 'Live Match', 'week_tag': st.session_state.saved_ladder_wk,
                                'is_popup': False
                            })
                    if valid_matches:
                        process_matches(valid_matches, name_to_id, df_players, df_leagues, df_meta)
                        st.success("Matches Saved to Database!")
                        
                        round_stats = {} 
                        all_names = st.session_state.ladder_live_roster['name'].unique()
                        for n in all_names: round_stats[n] = {'w':0, 'diff':0, 'pts':0}
                        
                        for r in valid_matches:
                            win = r['s1'] > r['s2']
                            diff = abs(r['s1'] - r['s2'])
                            for p in [r['t1_p1'], r['t1_p2']]:
                                round_stats[p]['pts'] += r['s1']
                                round_stats[p]['diff'] += diff if win else -diff
                                if win: round_stats[p]['w'] += 1
                            for p in [r['t2_p1'], r['t2_p2']]:
                                round_stats[p]['pts'] += r['s2']
                                round_stats[p]['diff'] += -diff if win else diff
                                if not win: round_stats[p]['w'] += 1
                        
                        df_roster = st.session_state.ladder_live_roster.copy()
                        df_roster['Round Wins'] = df_roster['name'].map(lambda x: round_stats.get(x, {}).get('w', 0))
                        df_roster['Round Diff'] = df_roster['name'].map(lambda x: round_stats.get(x, {}).get('diff', 0))
                        df_roster['Round Pts'] = df_roster['name'].map(lambda x: round_stats.get(x, {}).get('pts', 0))
                        
                        df_roster = df_roster.sort_values(by=['court', 'Round Wins', 'Round Diff', 'Round Pts'], ascending=[True, False, False, False])
                        df_roster['Proposed Court'] = df_roster['court']
                        
                        for c_num in sorted(df_roster['court'].unique()):
                            court_group = df_roster[df_roster['court'] == c_num]
                            if len(court_group) == 0: continue
                            top_player = court_group.iloc[0]['name']
                            btm_player = court_group.iloc[-1]['name']
                            if c_num > 1: df_roster.loc[df_roster['name'] == top_player, 'Proposed Court'] = c_num - 1
                            if c_num < len(st.session_state.ladder_court_sizes): df_roster.loc[df_roster['name'] == btm_player, 'Proposed Court'] = c_num + 1
                        
                        st.session_state.ladder_movement_preview = df_roster
                        st.session_state.ladder_state = 'CONFIRM_MOVEMENT'
                        del st.session_state.current_schedule
                        st.rerun()

        # 5. CONFIRM MOVEMENT
        if st.session_state.ladder_state == 'CONFIRM_MOVEMENT':
            st.markdown("#### Round Results & Movement")
            
            # --- VISUAL CARDS ---
            preview_df = st.session_state.ladder_movement_preview.sort_values('court')
            
            for c_num in sorted(preview_df['court'].unique()):
                st.markdown(f"<div class='court-header'>Court {c_num} Results</div>", unsafe_allow_html=True)
                c_players = preview_df[preview_df['court'] == c_num]
                
                # Show as a clean table with arrows
                for _, p in c_players.iterrows():
                    move_icon = "➖"
                    move_class = "move-stay"
                    if p['Proposed Court'] < p['court']: 
                        move_icon = f"🟢 ⬆️ To Ct {p['Proposed Court']}"
                        move_class = "move-up"
                    elif p['Proposed Court'] > p['court']: 
                        move_icon = f"🔴 ⬇️ To Ct {p['Proposed Court']}"
                        move_class = "move-down"
                    
                    col1, col2, col3 = st.columns([2, 2, 2])
                    col1.markdown(f"**{p['name']}**")
                    col2.markdown(f"W: {p['Round Wins']} | Diff: {p['Round Diff']}")
                    col3.markdown(f"<span class='{move_class}'>{move_icon}</span>", unsafe_allow_html=True)
                st.divider()

            st.markdown("#### 🛠️ Manual Override")
            st.info("Check the arrows above. If everything looks good, just click 'Start Next Round'. Otherwise, edit the 'New Ct' below.")
            
            editor_df = st.data_editor(
                st.session_state.ladder_movement_preview[['name', 'court', 'Round Wins', 'Round Diff', 'Proposed Court']],
                column_config={
                    "court": st.column_config.NumberColumn("Old Ct", disabled=True),
                    "Proposed Court": st.column_config.NumberColumn("New Ct", min_value=1, max_value=10, step=1)
                },
                hide_index=True,
                use_container_width=True
            )
            
            btn_label = "Start Next Round"
            if st.session_state.ladder_round_num >= st.session_state.ladder_total_rounds:
                btn_label = "🏁 Finish League Night"

            if st.button(btn_label):
                if st.session_state.ladder_round_num >= st.session_state.ladder_total_rounds:
                    st.balloons()
                    st.success("League Night Complete! All matches saved.")
                    time.sleep(3)
                    st.session_state.ladder_state = 'SETUP' # Reset
                    st.rerun()
                else:
                    new_roster = editor_df.copy()
                    new_roster['court'] = new_roster['Proposed Court']
                    new_roster = new_roster.sort_values('court')
                    new_sizes = new_roster['court'].value_counts().sort_index().tolist()
                    st.session_state.ladder_court_sizes = new_sizes
                    st.session_state.ladder_live_roster = new_roster[['name', 'court']] 
                    st.session_state.ladder_round_num += 1
                    st.session_state.ladder_state = 'PLAY_ROUND'
                    st.rerun()

    # --- TAB 2: SETTINGS ---
    with tabs[1]:
        st.subheader("Settings")
        if not df_meta.empty:
            editor = st.data_editor(df_meta[['id','league_name','is_active','min_games','description','k_factor']], disabled=['id','league_name'], hide_index=True, use_container_width=True)
            if st.button("Save Config"):
                for _,r in editor.iterrows(): supabase.table("leagues_metadata").update({"is_active":r['is_active'],"min_games":r['min_games'],"k_factor":r['k_factor'],"description":r['description']}).eq("id",r['id']).execute()
                st.rerun()
        
        # --- NEW DANGER ZONE ---
        st.divider()
        st.subheader("🗑️ Danger Zone")
        if not df_meta.empty:
            del_lg = st.selectbox("Select League to Delete", [""] + sorted(df_meta['league_name'].tolist()))
            if del_lg:
                st.error(f"⚠️ Are you sure you want to delete '{del_lg}'? This removes it from menus but keeps match history.")
                if st.button(f"Permanently Delete '{del_lg}'", type="primary"):
                    supabase.table("leagues_metadata").delete().eq("league_name", del_lg).eq("club_id", CLUB_ID).execute()
                    st.success(f"Deleted {del_lg}!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        st.write("#### 🆕 Create New League")
        with st.form("new"):
            n=st.text_input("Name"); k=st.number_input("K",32); mg=st.number_input("Min Games",12)
            if st.form_submit_button("Create"):
                supabase.table("leagues_metadata").insert({"club_id":CLUB_ID,"league_name":n,"is_active":True,"min_games":mg,"k_factor":k}).execute(); st.rerun()

elif sel == "📝 Match Uploader":
    st.header("📝 Match Uploader (Quick/Pop-Up)")
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

    week_tag = c3.selectbox("Week / Session", [f"Week {i}" for i in range(1, 13)] + ["Playoffs", "Finals", "Event"])
    st.divider()
    entry_method = st.radio("Entry Method", ["📋 Manual / Batch", "🏟️ Single Round Robin"], horizontal=True)
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
                        'week_tag': week_tag
                    })
            if valid_batch:
                process_matches(valid_batch, name_to_id, df_players, df_leagues, df_meta)
                st.success("✅ Processed!"); time.sleep(1); st.rerun()

    else:
        if 'lc_courts' not in st.session_state: st.session_state.lc_courts = 1
        st.session_state.lc_courts = st.number_input("Courts", 1, 10, st.session_state.lc_courts)
        with st.form("setup_lc"):
            c_data = []
            for i in range(st.session_state.lc_courts):
                c1, c2 = st.columns([1,3])
                t = c1.selectbox(f"Format C{i+1}", ["4-Player","5-Player","6-Player","8-Player","12-Player"], key=f"t_{i}")
                n = c2.text_area(f"Players C{i+1}", height=70, key=f"n_{i}")
                c_data.append({'type':t, 'names':n})
            
            st.markdown("---")
            custom_sched = st.text_area("Overrides: Paste Custom Schedule Here (e.g., '1 2 3 4')", help="Overrides the format selection above.")
            
            if st.form_submit_button("Generate"):
                st.session_state.lc_schedule = []
                st.session_state.active_lg = selected_league
                st.session_state.active_wk = week_tag
                st.session_state.active_is_popup = is_popup
                st.session_state.active_mt = match_type_db
                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.lc_schedule.append({'c': idx+1, 'm': get_match_schedule(c['type'], pl, custom_text=custom_sched)})
                st.rerun()
        
        if 'lc_schedule' in st.session_state:
            with st.form("scores_lc"):
                all_res = []
                for c in st.session_state.lc_schedule:
                    st.markdown(f"**Court {c['c']}**")
                    for i, m in enumerate(c['m']):
                        c1,c2,c3,c4 = st.columns([3,1,1,3])
                        c1.text(f"{m['t1'][0]}/{m['t1'][1]}"); s1=c2.number_input("S1",0,key=f"s1_{c['c']}_{i}"); s2=c3.number_input("S2",0,key=f"s2_{c['c']}_{i}"); c4.text(f"{m['t2'][0]}/{m['t2'][1]}")
                        all_res.append({'t1_p1':m['t1'][0],'t1_p2':m['t1'][1],'t2_p1':m['t2'][0],'t2_p2':m['t2'][1],'s1':s1,'s2':s2,'date':str(datetime.now()),'league':st.session_state.active_lg,'match_type':st.session_state.active_mt,'week_tag':st.session_state.active_wk, 'is_popup': st.session_state.active_is_popup})
                if st.form_submit_button("Submit"):
                     process_matches([x for x in all_res if x['s1']>0 or x['s2']>0], name_to_id, df_players, df_leagues, df_meta)
                     st.success("✅ Done!"); del st.session_state.lc_schedule; time.sleep(1); st.rerun()

elif sel == "👥 Player Editor":
    st.header("👥 Player Management")

    # --- TOP: ADD PLAYER ---
    with st.expander("➕ Add New Player", expanded=False):
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.5, step=0.1)
            if st.form_submit_button("Add Player"):
                safe_add_player(n, r)
                st.rerun()

    st.divider()

    # --- MAIN LAYOUT: LIST vs MANAGE ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Roster")
        # Display table sorted by name
        display_roster = df_players.sort_values("name")[['name', 'rating', 'wins', 'losses', 'matches_played']]
        # Convert rating for display
        display_roster['rating'] = display_roster['rating'] / 400
        st.dataframe(
            display_roster, 
            use_container_width=True, 
            hide_index=True,
            column_config={"rating": st.column_config.NumberColumn("JUPR", format="%.3f")}
        )

    with col2:
        st.subheader("Manage Player")
        p_edit = st.selectbox("Select Player to Edit/Delete", [""] + sorted(df_players['name'].tolist()))

        if p_edit:
            curr = df_players[df_players['name'] == p_edit].iloc[0]
            
            # EDIT FORM
            with st.form("edit_form"):
                st.caption(f"Editing: {p_edit}")
                new_n = st.text_input("Name", value=p_edit)
                new_r = st.number_input("Rating", 1.0, 7.0, float(curr['rating'])/400, step=0.01)
                
                if st.form_submit_button("Update Player"):
                    supabase.table("players").update({"name": new_n, "rating": new_r*400}).eq("id", int(curr['id'])).execute()
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()
            
            # DELETE BUTTON
            st.write("---")
            st.write("**Danger Zone**")
            if st.button("🗑️ Delete Player", type="primary"):
                # The new soft-delete way
                supabase.table("players").update({"active": False}).eq("id", int(curr['id'])).execute()
                st.success(f"Player {curr['name']} has been deactivated.")
                st.error(f"Deleted {p_edit}")
                time.sleep(1)
                st.rerun()

elif sel == "📝 Match Log":
    st.header("📝 Match Log")
    filter_type = st.radio("Filter", ["All", "League", "Pop-Up"], horizontal=True)
    if filter_type == "League": view_df = df_matches[df_matches['match_type'] != 'PopUp']
    elif filter_type == "Pop-Up": view_df = df_matches[df_matches['match_type'] == 'PopUp']
    else: view_df = df_matches
    
    col1, col2 = st.columns([1, 4])
    id_filter = col1.number_input("Jump to ID:", min_value=0, value=0)
    if id_filter > 0: view_df = view_df[view_df['id'] == id_filter]

    st.write("### 🗑️ Bulk Delete")
    if not view_df.empty:
        edit_df = view_df.head(500)[['id', 'date', 'league', 'match_type', 'elo_delta', 'p1', 'p2', 'p3', 'p4', 'score_t1', 'score_t2']].copy()
        edit_df.insert(0, "Delete", False) 
        edited_log = st.data_editor(edit_df, column_config={"Delete": st.column_config.CheckboxColumn(default=False)}, hide_index=True, use_container_width=True)
        to_delete = edited_log[edited_log['Delete'] == True]
        if not to_delete.empty:
            if st.button(f"Delete {len(to_delete)} Matches"):
                supabase.table("matches").delete().in_("id", to_delete['id'].tolist()).execute()
                st.success("Deleted!"); time.sleep(1); st.rerun()

elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")
    st.subheader("🏥 System Health Check")
    if st.button("Run Diagnostics"):
        with st.status("Checking System...", expanded=True) as status:
            try:
                sample = supabase.table("matches").select("*").limit(1).execute()
                if sample.data:
                    keys = sample.data[0].keys()
                    if 't1_p1_r' in keys: st.success("✅ Snapshot columns exist.")
                    else: st.error("❌ Snapshot columns MISSING in Supabase.")
                else: st.warning("⚠️ No matches found.")
                null_snaps = supabase.table("matches").select("id", "t1_p1_r").is_("t1_p1_r", "null").limit(5000).execute()
                if len(null_snaps.data) > 0: st.error(f"❌ Found {len(null_snaps.data)} matches with EMPTY snapshots. Run Recalculate below.")
                else: st.success("✅ All matches have snapshot data.")
            except Exception as e: st.error(f"Error: {e}")
            status.update(label="Complete", state="complete")

    st.divider()
    st.subheader("🔄 Recalculate")
    target_reset = st.selectbox("League", ["ALL (Full System Reset)"] + sorted(df_leagues['league_name'].unique().tolist()) if not df_leagues.empty else ["ALL"])
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
                sr1,sr2,sr3,sr4 = gr(p1),gr(p2),gr(p3),gr(p4)
                do1,do2 = calculate_hybrid_elo((sr1+sr2)/2,(sr3+sr4)/2,s1,s2)
                win=s1>s2
                for pid,d,w in [(p1,do1,win),(p2,do1,win),(p3,do2,not win),(p4,do2,not win)]:
                    if pid: p_map[pid]['r']+=d; p_map[pid]['mp']+=1; p_map[pid]['w' if w else 'l']+=1
                er1,er2,er3,er4 = gr(p1),gr(p2),gr(p3),gr(p4)
                if m.get('match_type') != 'PopUp':
                    def gir(pid, lg): 
                        if (pid,lg) not in island_map: island_map[(pid,lg)]={'r':p_map[pid]['r'],'w':0,'l':0,'mp':0}
                        return island_map[(pid,lg)]['r']
                    ir1,ir2,ir3,ir4 = gir(p1,m['league']),gir(p2,m['league']),gir(p3,m['league']),gir(p4,m['league'])
                    di1,di2 = calculate_hybrid_elo((ir1+ir2)/2,(ir3+ir4)/2,s1,s2)
                    for pid,d,w in [(p1,di1,win),(p2,di1,win),(p3,di2,not win),(p4,di2,not win)]:
                        if pid: k=(pid,m['league']); island_map[k]['r']+=d; island_map[k]['mp']+=1; island_map[k]['w' if w else 'l']+=1
                matches_to_update.append({'id':m['id'], 'elo_delta':do1 if s1>s2 else do2, 't1_p1_r':sr1,'t1_p2_r':sr2,'t2_p1_r':sr3,'t2_p2_r':sr4, 't1_p1_r_end':er1,'t1_p2_r_end':er2,'t2_p1_r_end':er3,'t2_p2_r_end':er4})

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

elif sel == "📘 Admin Guide":
    st.header("📘 Admin Guide")
    st.markdown("""
    ### 🏟️ League Manager (Live Event)
    1.  **Setup:** Set "Total Rounds" (e.g. 5). Paste names.
    2.  **Seeding:** Adjust court sizes. 
    3.  **Play:** Enter scores. They save instantly.
    4.  **Movement:** Review the Green/Red arrows. The "New Ct" column is editable if you need to override.
    
    ### 📝 Match Uploader
    Use this for pop-up events or entering data from paper sheets later.
    """)

import streamlit as st
import sqlite3
import pandas as pd
import io
import math

# Try to import xlsxwriter for smart templates
try:
    import xlsxwriter
except ImportError:
    xlsxwriter = None

# --- CONFIGURATION ---
K_FACTOR = 32
DEFAULT_START_RATING = 3.00

# --- DATABASE FUNCTIONS ---
def init_db():
    conn = sqlite3.connect('pickleball_club.db')
    c = conn.cursor()
    
    # Matches Table
    c.execute('''CREATE TABLE IF NOT EXISTS matches 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                  league TEXT,
                  t1_p1 TEXT, t1_p2 TEXT, t2_p1 TEXT, t2_p2 TEXT, 
                  score_t1 INTEGER, score_t2 INTEGER, 
                  elo_change_t1 REAL, elo_change_t2 REAL,
                  match_type TEXT)''')
    
    # Players Table
    c.execute('''CREATE TABLE IF NOT EXISTS players 
                 (name TEXT PRIMARY KEY, elo REAL, starting_elo REAL, 
                  matches_played INTEGER, wins INTEGER, losses INTEGER)''')
    
    # Migration Check
    c.execute("PRAGMA table_info(matches)")
    cols = [i[1] for i in c.fetchall()]
    if 'match_type' not in cols: c.execute("ALTER TABLE matches ADD COLUMN match_type TEXT")
    
    conn.commit()
    return conn

def get_player_data(conn, name):
    c = conn.cursor()
    c.execute("SELECT elo FROM players WHERE name=?", (name,))
    result = c.fetchone()
    if result: return result[0]
    else:
        start_elo = DEFAULT_START_RATING * 400
        c.execute("INSERT INTO players VALUES (?, ?, ?, 0, 0, 0)", (name, start_elo, start_elo))
        conn.commit()
        return start_elo

def upsert_player_rating(conn, name, rating, wins=0, losses=0):
    c = conn.cursor()
    elo = rating * 400 
    matches = wins + losses
    c.execute("SELECT name FROM players WHERE name=?", (name,))
    if c.fetchone():
        c.execute("""UPDATE players 
                     SET elo=?, starting_elo=?, wins=?, losses=?, matches_played=? 
                     WHERE name=?""", 
                  (elo, elo, wins, losses, matches, name))
    else:
        c.execute("INSERT INTO players VALUES (?, ?, ?, ?, ?, ?)", 
                  (name, elo, elo, matches, wins, losses))
    conn.commit()

def update_player_stats(conn, name, elo_delta, is_win):
    c = conn.cursor()
    win_inc = 1 if is_win else 0
    c.execute("""UPDATE players 
                 SET elo = elo + ?, matches_played = matches_played + 1, wins = wins + ?
                 WHERE name=?""", (elo_delta, win_inc, name))
    conn.commit()

# --- RECALCULATION ENGINE ---
def recalculate_history(conn):
    status_text = st.empty()
    status_text.info("⏳ Rewinding time... Resetting all player stats...")
    c = conn.cursor()
    c.execute("UPDATE players SET elo = starting_elo, wins = 0, losses = 0, matches_played = 0")
    matches = pd.read_sql_query("SELECT * FROM matches ORDER BY date ASC, id ASC", conn)
    total = len(matches)
    if total > 0:
        bar = st.progress(0)
        for i, m in matches.iterrows():
            p1, p2, p3, p4 = m['t1_p1'], m['t1_p2'], m['t2_p1'], m['t2_p2']
            s1, s2 = m['score_t1'], m['score_t2']
            r1, r2 = get_player_data(conn, p1), get_player_data(conn, p2)
            r3, r4 = get_player_data(conn, p3), get_player_data(conn, p4)
            dt1, dt2 = calculate_hybrid_elo((r1+r2)/2, (r3+r4)/2, s1, s2)
            win = s1 > s2
            update_player_stats(conn, p1, dt1, win)
            update_player_stats(conn, p2, dt1, win)
            update_player_stats(conn, p3, dt2, not win)
            update_player_stats(conn, p4, dt2, not win)
            c.execute("UPDATE matches SET elo_change_t1=?, elo_change_t2=? WHERE id=?", (dt1, dt2, m['id']))
            bar.progress((i + 1) / total)
    conn.commit()
    status_text.success("✅ History Replayed! All JUPR scores are now mathematically perfect.")
    st.balloons()

# --- MATH ENGINES ---
def calculate_hybrid_elo(t1_avg, t2_avg, score_t1, score_t2):
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1
    total_points = score_t1 + score_t2
    if total_points == 0: return 0, 0 
    actual_t1 = score_t1 / total_points
    actual_t2 = score_t2 / total_points
    raw_delta_t1 = K_FACTOR * 2 * (actual_t1 - expected_t1)
    raw_delta_t2 = K_FACTOR * 2 * (actual_t2 - expected_t2) 
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
    else:
        final_delta_t1 = raw_delta_t1
        final_delta_t2 = raw_delta_t2
    return final_delta_t1, final_delta_t2

# --- HELPER: HEADER SEARCH ---
def find_header_row(file_obj, candidate_cols=['PlayerID', 'Full Name', 'First', 'Last', 'Player']):
    file_obj.seek(0)
    for i in range(10):
        try:
            df = pd.read_csv(file_obj, header=i, nrows=1)
            if any(col in df.columns for col in candidate_cols):
                file_obj.seek(0)
                return i
        except: pass
    file_obj.seek(0)
    return 0

# --- HELPER: LEAGUE STATS ---
def get_league_leaderboard(conn, selected_league):
    df_players = pd.read_sql_query("SELECT name, elo, starting_elo FROM players", conn)
    if selected_league == "All":
        df_stats = pd.read_sql_query("SELECT name, wins, losses, matches_played FROM players", conn)
        merged = pd.merge(df_players, df_stats, on="name")
    else:
        matches_query = "SELECT * FROM matches WHERE league = ?"
        df_matches = pd.read_sql_query(matches_query, conn, params=(selected_league,))
        if df_matches.empty: return pd.DataFrame()
        stats = {}
        for _, row in df_matches.iterrows():
            s1, s2 = row['score_t1'], row['score_t2']
            t1 = [row['t1_p1'], row['t1_p2']]
            t2 = [row['t2_p1'], row['t2_p2']]
            def add_stat(p, is_win):
                if not p: return
                p = p.strip()
                if p not in stats: stats[p] = {'wins': 0, 'losses': 0}
                if is_win: stats[p]['wins'] += 1
                else: stats[p]['losses'] += 1
            if s1 > s2: 
                for p in t1: add_stat(p, True)
                for p in t2: add_stat(p, False)
            elif s2 > s1: 
                for p in t1: add_stat(p, False)
                for p in t2: add_stat(p, True)
        if not stats: return pd.DataFrame()
        stats_data = []
        for name, data in stats.items():
            stats_data.append({'name': name, 'wins': data['wins'], 'losses': data['losses'], 'matches_played': data['wins'] + data['losses']})
        df_league_stats = pd.DataFrame(stats_data)
        merged = pd.merge(df_players, df_league_stats, on="name", how="inner")

    merged['JUPR'] = (merged['elo'] / 400).map('{:,.3f}'.format)
    merged['Win %'] = (merged['wins'] / merged['matches_played'] * 100).fillna(0).map('{:.1f}%'.format)
    merged['Improvement'] = ((merged['elo'] - merged['starting_elo']) / 400).map('{:+.3f}'.format)
    return merged.sort_values(by='elo', ascending=False)

# --- HELPER: ROTATION ---
def get_match_schedule(court_type, players):
    matches = []
    p = players + [""] * (5 - len(players))
    if court_type == "4-Player":
        matches = [
            {'t1': [p[0], p[1]], 't2': [p[2], p[3]], 'desc': 'Round 1'},
            {'t1': [p[0], p[2]], 't2': [p[1], p[3]], 'desc': 'Round 2'},
            {'t1': [p[0], p[3]], 't2': [p[1], p[2]], 'desc': 'Round 3'}
        ]
    elif court_type == "5-Player":
        matches = [
            {'t1': [p[1], p[4]], 't2': [p[2], p[3]], 'desc': 'Round 1 (P1 Sits)'},
            {'t1': [p[0], p[4]], 't2': [p[1], p[2]], 'desc': 'Round 2 (P4 Sits)'},
            {'t1': [p[0], p[3]], 't2': [p[2], p[4]], 'desc': 'Round 3 (P2 Sits)'},
            {'t1': [p[0], p[1]], 't2': [p[3], p[4]], 'desc': 'Round 4 (P3 Sits)'},
            {'t1': [p[0], p[2]], 't2': [p[1], p[3]], 'desc': 'Round 5 (P5 Sits)'}
        ]
    return matches

# --- UI LAYOUT ---
st.set_page_config(page_title="Tres Palapas Pickleball", layout="wide")
st.title("🌵 Tres Palapas Pickleball Ratings and Ladder Results")

# --- CUSTOM CSS FOR FOOTER ---
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: #555;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    border-top: 1px solid #eee;
}
</style>
<div class="footer">
    <p>This data, program logic, and the "JUPR" Rating System are the intellectual property of <b>Joe Baumann</b>.</p>
</div>
""", unsafe_allow_html=True)

conn = init_db()

# --- JUPR EXPLANATION ---
with st.expander("ℹ️ About JUPR (Joe's Unique Pickleball Ratings)"):
    st.markdown("""
    **What is JUPR?**
    It stands for **Joe's Unique Pickleball Ratings**. It is a highly sophisticated, bespoke algorithm designed specifically for the chaos of ladder play.
    
    **How does it work?**
    It's complicated. But essentially:
    1.  **Expectations:** The system looks at who you are playing with and against. It calculates exactly how many points you *should* win based on everyone's history.
    2.  **Reality:** It compares that to what *actually* happened.
    3.  **The Secret Sauce:** If you outperform the computer's prediction, your JUPR goes up. If you underperform, it goes down. The magic is in the weighting—winning 11-0 against pros counts a lot more than winning 11-9 against beginners.
    
    *Trust the JUPR.*
    """)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Leaderboards", 
    "🏟️ Live Court Manager", 
    "👥 Player Management", 
    "📝 Match Log (Edit)"
])

# --- TAB 1: DYNAMIC LEADERBOARDS ---
with tab1:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.subheader("Filter")
        leagues = [r[0] for r in conn.execute("SELECT DISTINCT league FROM matches").fetchall()]
        if not leagues: leagues = ["Default"]
        leagues.insert(0, "All")
        selected_league = st.selectbox("Select League", leagues)

    with col_b:
        st.subheader(f"Standings: {selected_league}")
    
    display_df = get_league_leaderboard(conn, selected_league)
    
    if not display_df.empty:
        cols = ['name', 'JUPR', 'Improvement', 'matches_played', 'wins', 'losses', 'Win %']
        st.dataframe(display_df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No matches recorded for this league yet.")

# --- TAB 2: LIVE COURT MANAGER ---
with tab2:
    st.header("Live Court Manager")
    
    with st.expander("Setup Courts & Players", expanded=True):
        league_name = st.text_input("League / Event Name", "Fall 2025 Ladder")
        if 'num_courts' not in st.session_state: st.session_state.num_courts = 1
        st.session_state.num_courts = st.number_input("Count", 1, 20, st.session_state.num_courts)
        
        with st.form("court_setup"):
            court_data = []
            for i in range(st.session_state.num_courts):
                c1, c2 = st.columns([1, 4])
                with c1: ctype = st.selectbox(f"Format C{i+1}", ["4-Player", "5-Player"], key=f"fmt_{i}")
                with c2: names = st.text_area(f"Names C{i+1}", height=68, key=f"n_{i}", placeholder="Names separated by comma")
                court_data.append({'id': i+1, 'type': ctype, 'names': names})
            if st.form_submit_button("Generate Schedule"):
                st.session_state.schedule = []
                valid = True
                for c in court_data:
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    req = 4 if c['type'] == "4-Player" else 5
                    if len(pl) < req: 
                        st.error(f"Court {c['id']} needs {req} players"); valid=False
                    else:
                        st.session_state.schedule.append({'court': c['id'], 'matches': get_match_schedule(c['type'], pl)})
    
    if st.session_state.get('schedule'):
        st.divider()
        st.subheader("📝 Scorecard")
        with st.form("scores"):
            for c_idx, item in enumerate(st.session_state.schedule):
                st.markdown(f"**Court {item['court']}**")
                for m_idx, m in enumerate(item['matches']):
                    c1,c2,c3,c4 = st.columns([3,1,1,3])
                    with c1: st.text(f"{m['desc']}\n{m['t1'][0]} & {m['t1'][1]}")
                    with c2: s1 = st.number_input("S1", 0, key=f"s_{c_idx}_{m_idx}_1", label_visibility="collapsed")
                    with c3: s2 = st.number_input("S2", 0, key=f"s_{c_idx}_{m_idx}_2", label_visibility="collapsed")
                    with c4: st.text(f"\n{m['t2'][0]} & {m['t2'][1]}")
                    st.divider()
            
            if st.form_submit_button("Submit All"):
                count = 0
                for c_idx, item in enumerate(st.session_state.schedule):
                    for m_idx, m in enumerate(item['matches']):
                        s1 = st.session_state.get(f"s_{c_idx}_{m_idx}_1", 0)
                        s2 = st.session_state.get(f"s_{c_idx}_{m_idx}_2", 0)
                        if s1==0 and s2==0: continue
                        
                        p1,p2,p3,p4 = m['t1'][0], m['t1'][1], m['t2'][0], m['t2'][1]
                        r1,r2 = get_player_data(conn,p1), get_player_data(conn,p2)
                        r3,r4 = get_player_data(conn,p3), get_player_data(conn,p4)
                        dt1, dt2 = calculate_hybrid_elo((r1+r2)/2, (r3+r4)/2, s1, s2)
                        win = s1 > s2
                        
                        update_player_stats(conn, p1, dt1, win)
                        update_player_stats(conn, p2, dt1, win)
                        update_player_stats(conn, p3, dt2, not win)
                        update_player_stats(conn, p4, dt2, not win)
                        
                        c = conn.cursor()
                        c.execute("INSERT INTO matches (date, league, t1_p1, t1_p2, t2_p1, t2_p2, score_t1, score_t2, elo_change_t1, elo_change_t2, match_type) VALUES (datetime('now'),?,?,?,?,?,?,?,?,?,?)",
                                  (league_name, p1,p2,p3,p4,s1,s2,dt1,dt2, f"Court {item['court']} RR"))
                        count += 1
                conn.commit()
                st.success(f"Saved {count} matches!")
                del st.session_state.schedule
                st.rerun()

    with st.expander("Other Input Methods"):
        entry_method = st.radio("Method", ["Ladder Week Summary", "Manual Single"], horizontal=True)
        if entry_method == "Ladder Week Summary":
            f = st.file_uploader("Upload Week Template")
            if f: st.info("Please use Live Court Manager or Manual Single for best JUPR accuracy.")

# --- TAB 3: PLAYERS ---
with tab3:
    st.header("Player Management")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Add / Edit Player")
        with st.form("new_p"):
            n = st.text_input("Name (Exact Spelling)")
            r = st.number_input("Correct Starting JUPR", 1.0, 8.0, 3.0, 0.01)
            if st.form_submit_button("Save / Update Player"):
                upsert_player_rating(conn, n.strip(), r)
                st.success(f"Updated {n}. Start JUPR set to {r}. Go to Tab 4 to Recalculate History if needed.")
    
    with c2:
        st.subheader("Import Roster")
        f = st.file_uploader("Upload OverallResults.csv", type=['csv'])
        if f:
            h = find_header_row(f, ['Overall PB Rating'])
            df = pd.read_csv(f, header=h)
            if st.button("Import"):
                cnt = 0
                for _, row in df.iterrows():
                    try:
                        nm = str(row.iloc[1]).strip()
                        rt = float(row.iloc[2])
                        if nm and nm != 'nan':
                            upsert_player_rating(conn, nm, rt)
                            cnt += 1
                    except: pass
                st.success(f"Imported {cnt}")

# --- TAB 4: MATCH LOG (EDITABLE) ---
with tab4:
    st.header("Match Log & Editor")
    st.info("1. Edit rows to fix typos/scores. 2. Click 'Save & Recalculate' to fix JUPR.")
    
    df_log = pd.read_sql_query("SELECT id, date, league, t1_p1, t1_p2, score_t1, score_t2, t2_p1, t2_p2 FROM matches ORDER BY date DESC", conn)
    edited_df = st.data_editor(df_log, num_rows="dynamic", use_container_width=True)
    
    if st.button("💾 Save Changes & Recalculate JUPR"):
        try:
            c = conn.cursor()
            for _, row in edited_df.iterrows():
                if pd.notna(row['id']):
                    c.execute("""UPDATE matches 
                                 SET league=?, t1_p1=?, t1_p2=?, score_t1=?, score_t2=?, t2_p1=?, t2_p2=? 
                                 WHERE id=?""", 
                              (row['league'], row['t1_p1'], row['t1_p2'], row['score_t1'], 
                               row['score_t2'], row['t2_p1'], row['t2_p2'], row['id']))
            conn.commit()
            recalculate_history(conn)
        except Exception as e:
            st.error(f"Error updating: {e}")
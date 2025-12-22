import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import io

# --- CONFIGURATION ---
K_FACTOR = 32
DEFAULT_START_RATING = 3.00
SHEET_NAME = "Tres Palapas DB"  # Make sure this matches your Google Sheet name exactly

# --- GOOGLE SHEETS CONNECTION ---
def get_db_connection():
    """Connects to Google Sheets using Streamlit Secrets"""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # We load credentials from Streamlit secrets
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    try:
        sheet = client.open(SHEET_NAME)
        return sheet
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"❌ Could not find Google Sheet named '{SHEET_NAME}'. Did you share it with the service account email?")
        st.stop()

# --- DATA LOADERS (CACHED) ---
# We don't cache these heavily because we want instant updates for the admin
def load_data():
    sh = get_db_connection()
    try:
        players_ws = sh.worksheet("Players")
        matches_ws = sh.worksheet("Matches")
        
        # 1. LOAD PLAYERS safely
        p_data = players_ws.get_all_records()
        df_players = pd.DataFrame(p_data)
        
        # SAFETY CHECK: If sheet is empty or missing 'elo', force the columns to exist
        expected_p_cols = ['name', 'elo', 'starting_elo', 'matches_played', 'wins', 'losses']
        if df_players.empty or 'elo' not in df_players.columns:
            df_players = pd.DataFrame(columns=expected_p_cols)
        
        # 2. LOAD MATCHES safely
        m_data = matches_ws.get_all_records()
        df_matches = pd.DataFrame(m_data)
        
        expected_m_cols = ['score_t1', 'score_t2', 'elo_change_t1', 'elo_change_t2', 'league']
        if df_matches.empty:
             # Create minimal columns needed to prevent crashes
            df_matches = pd.DataFrame(columns=expected_m_cols)

        # 3. CLEAN UP NUMBERS (Convert text to numbers)
        for c in expected_p_cols:
            if c in df_players.columns and c != 'name': 
                df_players[c] = pd.to_numeric(df_players[c], errors='coerce').fillna(0)
            
        for c in expected_m_cols:
            if c in df_matches.columns and 'league' not in c:
                df_matches[c] = pd.to_numeric(df_matches[c], errors='coerce').fillna(0)
            
        return df_players, df_matches, players_ws, matches_ws
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()


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

# --- BATCH CALCULATOR (IN MEMORY) ---
def recalculate_all_stats(df_players, df_matches):
    """Replays history in memory and returns updated DataFrames"""
    
    # Reset Stats
    df_players['elo'] = df_players['starting_elo']
    df_players['wins'] = 0
    df_players['losses'] = 0
    df_players['matches_played'] = 0
    
    # Sort matches chronologically
    if not df_matches.empty:
        df_matches['date'] = pd.to_datetime(df_matches['date'])
        df_matches = df_matches.sort_values(by='date')
        
        # Replay
        for idx, row in df_matches.iterrows():
            p1, p2 = row['t1_p1'], row['t1_p2']
            p3, p4 = row['t2_p1'], row['t2_p2']
            s1, s2 = row['score_t1'], row['score_t2']
            
            # Lookup Ratings
            def get_elo(p_name):
                match = df_players[df_players['name'] == p_name]
                if match.empty: return DEFAULT_START_RATING * 400
                return match.iloc[0]['elo']
            
            r1, r2, r3, r4 = get_elo(p1), get_elo(p2), get_elo(p3), get_elo(p4)
            
            dt1, dt2 = calculate_hybrid_elo((r1+r2)/2, (r3+r4)/2, s1, s2)
            
            # Update Memory
            def update_p(p_name, delta, win):
                mask = df_players['name'] == p_name
                if not mask.any(): return # Skip if player deleted
                df_players.loc[mask, 'elo'] += delta
                df_players.loc[mask, 'matches_played'] += 1
                if win: df_players.loc[mask, 'wins'] += 1
                else: df_players.loc[mask, 'losses'] += 1
                
            win = s1 > s2
            update_p(p1, dt1, win)
            update_p(p2, dt1, win)
            update_p(p3, dt2, not win)
            update_p(p4, dt2, not win)
            
            # Update Match Record
            df_matches.at[idx, 'elo_change_t1'] = dt1
            df_matches.at[idx, 'elo_change_t2'] = dt2
            
    return df_players, df_matches

# --- UI LAYOUT ---
st.set_page_config(page_title="Tres Palapas Pickleball", layout="wide")
st.title("🌵 Tres Palapas Pickleball Ratings and Ladder Results")

# --- CUSTOM CSS FOR FOOTER ---
st.markdown("""
<style>
.footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: white; color: #555; text-align: center; padding: 10px; font-size: 12px; border-top: 1px solid #eee;}
</style>
<div class="footer"><p>This data, program logic, and the "JUPR" Rating System are the intellectual property of <b>Joe Baumann</b>.</p></div>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
df_players, df_matches, ws_players, ws_matches = load_data()

# --- LOGIN SYSTEM ---
if 'admin_logged_in' not in st.session_state: st.session_state.admin_logged_in = False

def check_password():
    if st.session_state.password == st.secrets["admin_password"]:
        st.session_state.admin_logged_in = True
    else:
        st.error("Incorrect Password")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Leaderboards", 
    "🏟️ Live Court Manager (Admin)", 
    "🔄 Other Round Robins (Admin)",
    "👥 Player Management (Admin)", 
    "📝 Match Log (Admin)"
])

# --- TAB 1: LEADERBOARDS (PUBLIC) ---
# --- TAB 1: LEADERBOARDS (PUBLIC) ---
with tab1:
    col_a, col_b = st.columns([1, 3])
    
    # 1. Get all available leagues
    leagues = ["All"]
    if not df_matches.empty:
        leagues += list(df_matches['league'].unique())
    
    # 2. Check URL for specific league (e.g. ?league=Fall2025)
    query_params = st.query_params
    default_index = 0
    
    if "league" in query_params:
        target_league = query_params["league"]
        if target_league in leagues:
            default_index = leagues.index(target_league)

    with col_a:
        st.subheader("Filter")
        # 3. Create the dropdown with the smart default
        selected_league = st.selectbox("Select League", leagues, index=default_index)
        
        # 4. Update the URL silently when they change the dropdown
        if selected_league != "All":
            st.query_params["league"] = selected_league
        else:
            # Clear the param if they select "All"
             if "league" in st.query_params: 
                 del st.query_params["league"]

    with col_b:
        st.subheader(f"Standings: {selected_league}")
        # 5. Show a shareable link for this specific league
        if selected_league != "All":
            base_url = "https://8lkemld946rmtwwptk2gcs.streamlit.app/" # REPLACE THIS WITH YOUR ACTUAL URL
            share_link = f"{base_url}?league={selected_league.replace(' ', '%20')}"
            st.code(share_link, language="text")
            st.caption("Copy this link to send players directly to this league.")
    
    # Calculate League Specific Stats
    display_df = df_players.copy()
    
    if selected_league != "All" and not df_matches.empty:
        # Filter matches
        league_matches = df_matches[df_matches['league'] == selected_league]
        
        # Re-tally wins/losses just for this view (Ratings stay global)
        stats = {}
        for _, row in league_matches.iterrows():
            s1, s2 = row['score_t1'], row['score_t2']
            winners = [row['t1_p1'], row['t1_p2']] if s1 > s2 else [row['t2_p1'], row['t2_p2']]
            losers = [row['t2_p1'], row['t2_p2']] if s1 > s2 else [row['t1_p1'], row['t1_p2']]
            
            for p in winners: 
                if p not in stats: stats[p] = {'w':0, 'l':0}
                stats[p]['w'] += 1
            for p in losers:
                if p not in stats: stats[p] = {'w':0, 'l':0}
                stats[p]['l'] += 1
        
        # Map back to display_df
        display_df['wins'] = display_df['name'].map(lambda x: stats.get(x, {'w':0})['w'])
        display_df['losses'] = display_df['name'].map(lambda x: stats.get(x, {'l':0})['l'])
        display_df['matches_played'] = display_df['wins'] + display_df['losses']
        display_df = display_df[display_df['matches_played'] > 0]

    # Format
    display_df['JUPR'] = (display_df['elo'] / 400).map('{:,.3f}'.format)
    display_df['Win %'] = (display_df['wins'] / display_df['matches_played'] * 100).fillna(0).map('{:.1f}%'.format)
    display_df['Improvement'] = ((display_df['elo'] - display_df['starting_elo']) / 400).map('{:+.3f}'.format)
    
    # CORRECT LOGIC: Sort by 'elo' FIRST
    sorted_df = display_df.sort_values(by='elo', ascending=False)

    # THEN select only the columns you want to show
    st.dataframe(
        sorted_df[['name', 'JUPR', 'Improvement', 'matches_played', 'wins', 'losses', 'Win %']], 
        use_container_width=True, 
        hide_index=True
    )

    with col_b:
        st.subheader(f"Standings: {selected_league}")
    
    # Calculate League Specific Stats
    display_df = df_players.copy()
    
    if selected_league != "All" and not df_matches.empty:
        # Filter matches
        league_matches = df_matches[df_matches['league'] == selected_league]
        
        # Re-tally wins/losses just for this view (Ratings stay global)
        stats = {}
        for _, row in league_matches.iterrows():
            s1, s2 = row['score_t1'], row['score_t2']
            winners = [row['t1_p1'], row['t1_p2']] if s1 > s2 else [row['t2_p1'], row['t2_p2']]
            losers = [row['t2_p1'], row['t2_p2']] if s1 > s2 else [row['t1_p1'], row['t1_p2']]
            
            for p in winners: 
                if p not in stats: stats[p] = {'w':0, 'l':0}
                stats[p]['w'] += 1
            for p in losers:
                if p not in stats: stats[p] = {'w':0, 'l':0}
                stats[p]['l'] += 1
        
        # Map back to display_df
        display_df['wins'] = display_df['name'].map(lambda x: stats.get(x, {'w':0})['w'])
        display_df['losses'] = display_df['name'].map(lambda x: stats.get(x, {'l':0})['l'])
        display_df['matches_played'] = display_df['wins'] + display_df['losses']
        display_df = display_df[display_df['matches_played'] > 0]

    # Format
    display_df['JUPR'] = (display_df['elo'] / 400).map('{:,.3f}'.format)
    display_df['Win %'] = (display_df['wins'] / display_df['matches_played'] * 100).fillna(0).map('{:.1f}%'.format)
    display_df['Improvement'] = ((display_df['elo'] - display_df['starting_elo']) / 400).map('{:+.3f}'.format)
    
  # CORRECT LOGIC: Sort by 'elo' FIRST, while it still exists
    sorted_df = display_df.sort_values(by='elo', ascending=False)

    # THEN select only the columns you want to show
    st.dataframe(
        sorted_df[['name', 'JUPR', 'Improvement', 'matches_played', 'wins', 'losses', 'Win %']], 
        use_container_width=True, 
        hide_index=True
    )

# --- ADMIN GATEKEEPER ---
if not st.session_state.admin_logged_in:
    with tab2: st.warning("Admin Access Only"); st.text_input("Admin Password", type="password", key="password", on_change=check_password)
    with tab3: st.warning("Admin Access Only")
    with tab4: st.warning("Admin Access Only")
    with tab5: st.warning("Admin Access Only")
    st.stop() # Stop rendering the rest

# --- HELPER: SCHEDULE GENERATOR ---
def get_match_schedule(court_type, players):
    p = players + ["?"] * (12 - len(players))
    matches = []
    
    if court_type == "4-Player":
        matches = [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif court_type == "5-Player":
        matches = [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1 (1 Sit)'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2 (4 Sit)'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3 (2 Sit)'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4 (3 Sit)'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5 (5 Sit)'}]
    elif court_type == "6-Player":
        matches = [{'t1':[p[0],p[1]],'t2':[p[2],p[4]],'desc':'R1 (4,6 Sit)'}, {'t1':[p[2],p[5]],'t2':[p[0],p[4]],'desc':'R2 (1,2 Sit)'}, {'t1':[p[1],p[3]],'t2':[p[4],p[5]],'desc':'R3 (1,3 Sit)'}, {'t1':[p[0],p[5]],'t2':[p[1],p[2]],'desc':'R4 (3,4 Sit)'}, {'t1':[p[0],p[3]],'t2':[p[1],p[4]],'desc':'R5 (2,5 Sit)'}]
    elif court_type == "8-Player":
        matches = [
            {'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1 A'}, {'t1':[p[4],p[5]],'t2':[p[6],p[7]],'desc':'R1 B'},
            {'t1':[p[0],p[2]],'t2':[p[4],p[6]],'desc':'R2 A'}, {'t1':[p[1],p[3]],'t2':[p[5],p[7]],'desc':'R2 B'},
            {'t1':[p[0],p[3]],'t2':[p[5],p[6]],'desc':'R3 A'}, {'t1':[p[1],p[2]],'t2':[p[4],p[7]],'desc':'R3 B'},
            {'t1':[p[0],p[4]],'t2':[p[1],p[5]],'desc':'R4 A'}, {'t1':[p[2],p[6]],'t2':[p[3],p[7]],'desc':'R4 B'}
        ]
    # Add 9 and 12 logic as previously defined if needed
    return matches

# --- TAB 2: LIVE COURT MANAGER ---
with tab2:
    st.header("Live Court Manager")
    with st.expander("Setup", expanded=True):
        league_name = st.text_input("League", "Fall 2025 Ladder")
        num_courts = st.number_input("Courts", 1, 20, 1)
        with st.form("setup"):
            court_data = []
            for i in range(num_courts):
                c1,c2 = st.columns([1,4])
                with c1: t = st.selectbox(f"Type {i+1}", ["4-Player","5-Player","6-Player","8-Player"], key=f"t{i}")
                with c2: n = st.text_area(f"Names {i+1}", key=f"n{i}", height=68)
                court_data.append({'id':i+1,'type':t,'names':n})
            if st.form_submit_button("Generate"):
                st.session_state.schedule = []
                for c in court_data:
                    pl = [x.strip() for x in c['names'].replace('\n',',').split(',') if x.strip()]
                    st.session_state.schedule.append({'court':c['id'], 'matches':get_match_schedule(c['type'], pl)})
    
    if st.session_state.get('schedule'):
        st.divider()
        with st.form("submit_scores"):
            for c in st.session_state.schedule:
                st.markdown(f"**Court {c['court']}**")
                for i, m in enumerate(c['matches']):
                    c1,c2,c3,c4 = st.columns([3,1,1,3])
                    with c1: st.text(f"{m['desc']} | {m['t1'][0]} & {m['t1'][1]}")
                    with c2: s1 = st.number_input("S1", 0, key=f"s_{c['court']}_{i}_1")
                    with c3: s2 = st.number_input("S2", 0, key=f"s_{c['court']}_{i}_2")
                    with c4: st.text(f"{m['t2'][0]} & {m['t2'][1]}")
            
            if st.form_submit_button("Submit & Save to Cloud"):
                new_matches = []
                for c in st.session_state.schedule:
                    for i, m in enumerate(c['matches']):
                        s1 = st.session_state.get(f"s_{c['court']}_{i}_1", 0)
                        s2 = st.session_state.get(f"s_{c['court']}_{i}_2", 0)
                        if s1 == 0 and s2 == 0: continue
                        
                        # Add to list
                        new_matches.append({
                            'id': len(df_matches) + len(new_matches) + 1,
                            'date': str(datetime.now()),
                            'league': league_name,
                            't1_p1': m['t1'][0], 't1_p2': m['t1'][1],
                            't2_p1': m['t2'][0], 't2_p2': m['t2'][1],
                            'score_t1': s1, 'score_t2': s2,
                            'elo_change_t1': 0, 'elo_change_t2': 0, # Calc later
                            'match_type': f"Court {c['court']} RR"
                        })
                
                # Append to dataframe
                if new_matches:
                    new_df = pd.DataFrame(new_matches)
                    df_matches = pd.concat([df_matches, new_df], ignore_index=True)
                    
                    # Recalc All Stats (To update ratings)
                    df_players, df_matches = recalculate_all_stats(df_players, df_matches)
                    
                    # WRITE TO GOOGLE SHEETS
                    ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
                    ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
                    
                    st.success("✅ Saved to Google Cloud!")
                    st.rerun()

# --- TAB 3: OTHER RR ---
with tab3:
    st.info("Standalone generator (Logic same as above)")

# --- TAB 4: PLAYERS ---
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Start JUPR", 3.0)
            if st.form_submit_button("Add Player"):
                if n not in df_players['name'].values:
                    new_p = {'name': n, 'elo': r*400, 'starting_elo': r*400, 'matches_played':0, 'wins':0, 'losses':0}
                    df_players = pd.concat([df_players, pd.DataFrame([new_p])], ignore_index=True)
                    ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
                    st.success("Added to Cloud")
                    st.rerun()
                else: st.error("Player exists")

# --- TAB 5: MATCH LOG ---
with tab5:
    st.header("Match Editor (Cloud)")
    st.info("Edits here update the Google Sheet directly.")
    
    edited_df = st.data_editor(df_matches, num_rows="dynamic", use_container_width=True)
    
    if st.button("💾 Save Changes & Recalc All History"):
        # 1. Update Match Data
        df_matches = edited_df
        # 2. Recalculate Ratings from Scratch
        df_players, df_matches = recalculate_all_stats(df_players, df_matches)
        # 3. Push to Cloud
        ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
        ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
        st.success("Cloud Updated!")

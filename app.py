import streamlit as st
import pandas as pd
import gspread
import math
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

# --- DATA LOADERS (CORRECTED) ---
def load_data():
    sh = get_db_connection()
    try:
        # Load the 3 Core Sheets
        players_ws = sh.worksheet("Players")       # Legacy / Metadata
        matches_ws = sh.worksheet("Matches")       # Match History
        ratings_ws = sh.worksheet("player_ratings") # NEW ISLAND RATINGS
        
        # 1. LOAD LEGACY PLAYERS (Required for Tab 4 & Roster checks)
        p_data = players_ws.get_all_records()
        df_players = pd.DataFrame(p_data)
        
        # Safety check for players
        expected_p_cols = ['name', 'elo', 'starting_elo', 'matches_played', 'wins', 'losses']
        if df_players.empty or 'name' not in df_players.columns:
            df_players = pd.DataFrame(columns=expected_p_cols)

        # 2. LOAD RATINGS (The New Engine)
        r_data = ratings_ws.get_all_records()
        df_ratings = pd.DataFrame(r_data)
        if df_ratings.empty:
            df_ratings = pd.DataFrame(columns=['name', 'ladder_id', 'rating'])

        # 3. LOAD MATCHES (History)
        m_data = matches_ws.get_all_records()
        df_matches = pd.DataFrame(m_data)
        
        expected_m_cols = ['score_t1', 'score_t2', 'league', 't1_p1', 't1_p2', 't2_p1', 't2_p2']
        if df_matches.empty:
             df_matches = pd.DataFrame(columns=expected_m_cols)

        # 4. Clean up Numbers
        for c in ['score_t1', 'score_t2']:
            if c in df_matches.columns:
                df_matches[c] = pd.to_numeric(df_matches[c], errors='coerce').fillna(0)
            
        # RETURN EVERYTHING (6 items)
        return df_players, df_ratings, df_matches, players_ws, matches_ws, ratings_ws

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

# --- ISLAND / LADDER LOGIC START ---

def calculate_ladder_elo(rating_a, rating_b, actual_score_a):
    """Standard 1v1 Elo for Ladder Matches."""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + K_FACTOR * (actual_score_a - expected_a)
    return round(new_rating_a)

def get_effective_rating(all_rows, user_name, ladder_id):
    """
    Finds rating for a specific ladder. 
    If not found, 'Seeds' from OVERALL. 
    If OVERALL not found, returns Default (1200).
    """
    # 1. Search for specific ladder rating
    for row in all_rows:
        if row['name'] == user_name and row['ladder_id'] == ladder_id:
            return float(row['rating'])
    
    # 2. If not found, look for OVERALL (Seeding)
    if ladder_id != 'OVERALL':
        for row in all_rows:
            if row['name'] == user_name and row['ladder_id'] == 'OVERALL':
                return float(row['rating'])
    
    # 3. Default start (1200 or whatever you prefer)
    return 1200.0

def update_local_memory(all_rows, name, ladder, new_rating):
    """Updates the list of dictionaries in memory."""
    found = False
    for row in all_rows:
        if row['name'] == name and row['ladder_id'] == ladder:
            row['rating'] = new_rating
            found = True
            break
    if not found:
        all_rows.append({'name': name, 'ladder_id': ladder, 'rating': new_rating})

def process_batch_upload(dataframe, ladder_name_from_ui=None):
    """
    Takes uploaded CSV, runs Island/Soft Seeding logic, and updates Google Sheet.
    """
    # 1. Use the EXISTING Streamlit connection (Secure)
    sh = get_db_connection()
    try:
        ratings_sheet = sh.worksheet("player_ratings")
    except gspread.exceptions.WorksheetNotFound:
        st.error("❌ Tab 'player_ratings' not found in Google Sheets. Please create it!")
        return []
    
    # 2. Load current state
    all_rows = ratings_sheet.get_all_records()
    results_log = []

    # 3. Process Rows
    for index, row in dataframe.iterrows():
        # Handle case-insensitive columns
        winner = row.get('winner') or row.get('Winner')
        loser = row.get('loser') or row.get('Loser')
        
        # Determine Ladder Name
        csv_ladder = row.get('ladder') or row.get('Ladder')
        ladder_id = csv_ladder if csv_ladder else ladder_name_from_ui
        
        if not winner or not loser or not ladder_id:
            continue # Skip invalid rows

        # --- UPDATE SPECIFIC LADDER ---
        r_win = get_effective_rating(all_rows, winner, ladder_id)
        r_lose = get_effective_rating(all_rows, loser, ladder_id)
        
        new_win = calculate_ladder_elo(r_win, r_lose, 1)
        new_lose = calculate_ladder_elo(r_lose, r_win, 0)
        
        update_local_memory(all_rows, winner, ladder_id, new_win)
        update_local_memory(all_rows, loser, ladder_id, new_lose)

        # --- UPDATE OVERALL LADDER ---
        r_ov_win = get_effective_rating(all_rows, winner, 'OVERALL')
        r_ov_lose = get_effective_rating(all_rows, loser, 'OVERALL')
        
        new_ov_win = calculate_ladder_elo(r_ov_win, r_ov_lose, 1)
        new_ov_lose = calculate_ladder_elo(r_ov_lose, r_ov_win, 0)
        
        update_local_memory(all_rows, winner, 'OVERALL', new_ov_win)
        update_local_memory(all_rows, loser, 'OVERALL', new_ov_lose)
        
        results_log.append(f"Processed: {winner} def. {loser} ({ladder_id})")

    # 4. Save back to Google Sheets (Bulk Update)
    if all_rows:
        headers = list(all_rows[0].keys())
        data_to_write = [headers] + [list(r.values()) for r in all_rows]
        ratings_sheet.clear()
        ratings_sheet.update(data_to_write)
        
    return results_log

# --- HISTORY REPLAY LOGIC ---
def replay_league_history(target_league):
    """
    Reads existing match history for a specific league and 
    re-calculates the Island Ratings from scratch.
    """
    sh = get_db_connection()
    ratings_sheet = sh.worksheet("player_ratings")
    all_rows = ratings_sheet.get_all_records()
    
    # 1. Load Match History
    matches_ws = sh.worksheet("Matches")
    m_data = matches_ws.get_all_records()
    df_history = pd.DataFrame(m_data)
    
    # Filter for the league we want to restore
    if 'league' not in df_history.columns:
        return f"Error: No 'league' column found in matches."
    
    # Sort by date to ensure correct order
    if 'date' in df_history.columns:
        df_history['date'] = pd.to_datetime(df_history['date'])
        df_history = df_history.sort_values(by='date')
    
    league_matches = df_history[df_history['league'] == target_league]
    
    if league_matches.empty:
        return f"No matches found for league: {target_league}"

    count = 0
    # 2. Replay every match
    for _, row in league_matches.iterrows():
        # Construct match data object from the row
        match_data = {
            't1_p1': row.get('t1_p1'), 't1_p2': row.get('t1_p2'),
            't2_p1': row.get('t2_p1'), 't2_p2': row.get('t2_p2'),
            'score_t1': row.get('score_t1'), 'score_t2': row.get('score_t2')
        }
        
        # Use the standard Island Processor
        # This will auto-seed from OVERALL if they don't have a rating yet
        process_live_doubles_match(match_data, ladder_name=target_league)
        count += 1

    return f"Successfully restored '{target_league}'! Processed {count} historical matches."

# --- DOUBLES ISLAND LOGIC ---
def process_live_doubles_match(match_data, ladder_name):
    """
    Handles a single 2v2 match from the Live Court Manager.
    Updates both the Specific Ladder and OVERALL ratings.
    """
    # 1. Connect to Sheet
    sh = get_db_connection()
    ratings_sheet = sh.worksheet("player_ratings")
    all_rows = ratings_sheet.get_all_records()

    # 2. Extract Data
    t1 = [match_data['t1_p1'], match_data['t1_p2']]
    t2 = [match_data['t2_p1'], match_data['t2_p2']]
    s1 = match_data['score_t1']
    s2 = match_data['score_t2']
    
    # 3. Determine Winner/Loser Teams
    if s1 > s2:
        winners, losers = t1, t2
        actual_score = 1
    else:
        winners, losers = t2, t1
        actual_score = 0 # From perspective of Team 1, but we calculate per team below

    # --- HELPER: UPDATE ONE LADDER CONTEXT ---
    def update_context(context_id):
        # A. Get Ratings
        w_ratings = [get_effective_rating(all_rows, p, context_id) for p in winners]
        l_ratings = [get_effective_rating(all_rows, p, context_id) for p in losers]
        
        # B. Calculate Team Averages
        w_avg = sum(w_ratings) / len(w_ratings)
        l_avg = sum(l_ratings) / len(l_ratings)
        
        # C. Calculate Elo Delta (Using Average)
        # We assume Winners won (score=1) against Losers
        expected_win = 1 / (1 + 10 ** ((l_avg - w_avg) / 400))
        delta = K_FACTOR * (1 - expected_win)
        
        # D. Apply Delta to INDIVIDUALS
        for p, r in zip(winners, w_ratings):
            update_local_memory(all_rows, p, context_id, round(r + delta))
            
        for p, r in zip(losers, l_ratings):
            update_local_memory(all_rows, p, context_id, round(r - delta))

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
    return matches

# --- OVERALL-ONLY LOGIC (FOR TAB 3) ---
def process_overall_only_match(match_data):
    """
    Handles a match that counts ONLY toward the Global Rating, not a specific ladder.
    """
    sh = get_db_connection()
    ratings_sheet = sh.worksheet("player_ratings")
    all_rows = ratings_sheet.get_all_records()

    t1 = [match_data['t1_p1'], match_data['t1_p2']]
    t2 = [match_data['t2_p1'], match_data['t2_p2']]
    s1 = match_data['score_t1']
    s2 = match_data['score_t2']

    if s1 > s2:
        winners, losers = t1, t2
    else:
        winners, losers = t2, t1

    # --- UPDATE ONLY THE 'OVERALL' CONTEXT ---
    context_id = "OVERALL"
    
    # A. Get Ratings
    w_ratings = [get_effective_rating(all_rows, p, context_id) for p in winners]
    l_ratings = [get_effective_rating(all_rows, p, context_id) for p in losers]
    
    # B. Calculate Team Averages
    w_avg = sum(w_ratings) / len(w_ratings)
    l_avg = sum(l_ratings) / len(l_ratings)
    
    # C. Calculate Elo Delta
    expected_win = 1 / (1 + 10 ** ((l_avg - w_avg) / 400))
    delta = K_FACTOR * (1 - expected_win)
    
    # D. Apply Delta
    for p, r in zip(winners, w_ratings):
        update_local_memory(all_rows, p, context_id, round(r + delta))
    for p, r in zip(losers, l_ratings):
        update_local_memory(all_rows, p, context_id, round(r - delta))

    # Save to Cloud
    if all_rows:
        headers = list(all_rows[0].keys())
        data_to_write = [headers] + [list(r.values()) for r in all_rows]
        ratings_sheet.clear()
        ratings_sheet.update(data_to_write)
    
    # 4. Run Updates
    # Update Specific Island (The League Name)
    update_context(ladder_name)
    
    # Update OVERALL
    update_context("OVERALL")

    # 5. Save to Cloud
    if all_rows:
        headers = list(all_rows[0].keys())
        data_to_write = [headers] + [list(r.values()) for r in all_rows]
        ratings_sheet.clear()
        ratings_sheet.update(data_to_write)

# --- ISLAND / LADDER LOGIC END ---

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
# We now unpack 6 items safely (Added df_players at the start)
df_players, df_ratings, df_matches, ws_players, ws_matches, ws_ratings = load_data()

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

# --- TAB 1: LEADERBOARDS (NEW ISLAND SYSTEM) ---
with tab1:
    col_a, col_b = st.columns([1, 3])
    
    # 1. Get List of Ladders (Islands)
    available_ladders = ["OVERALL"]
    if not df_ratings.empty:
        others = [x for x in df_ratings['ladder_id'].unique() if x != "OVERALL"]
        available_ladders += sorted(others)
    
    # 2. Check URL for sharing
    query_params = st.query_params
    default_index = 0
    if "league" in query_params:
        target = query_params["league"]
        if target in available_ladders:
            default_index = available_ladders.index(target)

    with col_a:
        st.subheader("Filter")
        selected_ladder = st.selectbox("Select Ladder", available_ladders, index=default_index)
        
        # Update URL
        if selected_ladder != "OVERALL":
            st.query_params["league"] = selected_ladder
        else:
            if "league" in st.query_params: del st.query_params["league"]

    with col_b:
        st.subheader(f"Standings: {selected_ladder}")
    
    # --- BUILD THE LEADERBOARD ---
    if not df_ratings.empty:
        # A. Filter Ratings for this specific Island
        ladder_ratings = df_ratings[df_ratings['ladder_id'] == selected_ladder].copy()
        
        # B. Calculate Win/Loss Record from Match History
        if selected_ladder == "OVERALL":
            relevant_matches = df_matches
        else:
            relevant_matches = df_matches[df_matches['league'] == selected_ladder]

        stats = {}
        for _, row in relevant_matches.iterrows():
            s1, s2 = row['score_t1'], row['score_t2']
            if s1 > s2:
                wins = [row['t1_p1'], row['t1_p2']]
                loss = [row['t2_p1'], row['t2_p2']]
            else:
                wins = [row['t2_p1'], row['t2_p2']]
                loss = [row['t1_p1'], row['t1_p2']]
            
            for p in wins:
                if p not in stats: stats[p] = {'w': 0, 'l': 0}
                stats[p]['w'] += 1
            for p in loss:
                if p not in stats: stats[p] = {'w': 0, 'l': 0}
                stats[p]['l'] += 1

        # C. Merge Stats into Ratings
        ladder_ratings['wins'] = ladder_ratings['name'].map(lambda x: stats.get(x, {'w':0})['w'])
        ladder_ratings['losses'] = ladder_ratings['name'].map(lambda x: stats.get(x, {'l':0})['l'])
        ladder_ratings['matches'] = ladder_ratings['wins'] + ladder_ratings['losses']
        
        # D. Format and Sort
        # THIS IS THE FIX: Divide by 400 to get JUPR format (e.g. 3.521)
        ladder_ratings['JUPR'] = (ladder_ratings['rating'] / 400).map('{:,.3f}'.format)
        
        # Win % Calculation
        ladder_ratings['Win %'] = (ladder_ratings['wins'] / ladder_ratings['matches'] * 100).fillna(0).map('{:.1f}%'.format)

        # Sort by Rating High -> Low
        leaderboard = ladder_ratings.sort_values(by='rating', ascending=False)
        
        # E. Display
        st.dataframe(
            leaderboard[['name', 'JUPR', 'matches', 'wins', 'losses', 'Win %']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No ratings found. Upload matches to generate the first leaderboard!")

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
                
                # 1. Gather all results from the form
                for c in st.session_state.schedule:
                    for i, m in enumerate(c['matches']):
                        s1 = st.session_state.get(f"s_{c['court']}_{i}_1", 0)
                        s2 = st.session_state.get(f"s_{c['court']}_{i}_2", 0)
                        
                        # Skip games with no score
                        if s1 == 0 and s2 == 0: continue
                        
                        # Create Match Object
                        match_data = {
                            'id': len(df_matches) + len(new_matches) + 1,
                            'date': str(datetime.now()),
                            'league': league_name, # This becomes the Island ID!
                            't1_p1': m['t1'][0], 't1_p2': m['t1'][1],
                            't2_p1': m['t2'][0], 't2_p2': m['t2'][1],
                            'score_t1': s1, 'score_t2': s2,
                            'match_type': f"Court {c['court']} RR"
                        }
                        
                        # 2. RUN THE NEW ISLAND LOGIC
                        process_live_doubles_match(match_data, ladder_name=league_name)
                        
                        # 3. Log it for history (Optional but recommended)
                        new_matches.append(match_data)
                
                # 4. Save the Match Log (Just for record keeping, not for math)
                if new_matches:
                    new_df = pd.DataFrame(new_matches)
                    df_matches = pd.concat([df_matches, new_df], ignore_index=True)
                    ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
                    
                    st.success(f"✅ Processed {len(new_matches)} matches into League '{league_name}'!")
                    st.rerun()

# --- TAB 3: POP-UP ROUND ROBINS (OVERALL RATING ONLY) ---
with tab3:
    st.header("Pop-Up Round Robin")
    st.caption("Matches played here affect **OVERALL RATING** but do not affect specific League Ladders.")
    
    with st.expander("Event Setup", expanded=True):
        # We give this a name for the history logs
        popup_name = st.text_input("Event Name", f"PopUp {datetime.now().strftime('%Y-%m-%d')}")
        rr_courts = st.number_input("Number of Courts", 1, 20, 1, key="rr_courts")
        
        with st.form("rr_setup"):
            rr_data = []
            for i in range(rr_courts):
                c1, c2 = st.columns([1, 4])
                with c1: 
                    t = st.selectbox(f"Format {i+1}", ["4-Player", "5-Player", "6-Player", "8-Player"], key=f"rr_t{i}")
                with c2: 
                    n = st.text_area(f"Names {i+1}", key=f"rr_n{i}", height=68, placeholder="Joe, Kevin, Scott, Robin...")
                rr_data.append({'id': i+1, 'type': t, 'names': n})
            
            if st.form_submit_button("Generate Schedule"):
                st.session_state.rr_schedule = []
                for c in rr_data:
                    pl = [x.strip() for x in c['names'].replace('\n', ',').split(',') if x.strip()]
                    st.session_state.rr_schedule.append({'court': c['id'], 'matches': get_match_schedule(c['type'], pl)})

    # Display Schedule & Score Inputs
    if st.session_state.get('rr_schedule'):
        st.divider()
        with st.form("submit_rr_scores"):
            for c in st.session_state.rr_schedule:
                st.markdown(f"**Court {c['court']}**")
                for i, m in enumerate(c['matches']):
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                    with c1: st.text(f"{m['desc']} | {m['t1'][0]} & {m['t1'][1]}")
                    with c2: s1 = st.number_input("S1", 0, key=f"rr_s_{c['court']}_{i}_1")
                    with c3: s2 = st.number_input("S2", 0, key=f"rr_s_{c['court']}_{i}_2")
                    with c4: st.text(f"{m['t2'][0]} & {m['t2'][1]}")
            
            if st.form_submit_button("Submit to Overall Ratings"):
                new_matches = []
                for c in st.session_state.rr_schedule:
                    for i, m in enumerate(c['matches']):
                        s1 = st.session_state.get(f"rr_s_{c['court']}_{i}_1", 0)
                        s2 = st.session_state.get(f"rr_s_{c['court']}_{i}_2", 0)
                        
                        if s1 == 0 and s2 == 0: continue
                        
                        match_data = {
                            'id': len(df_matches) + len(new_matches) + 1,
                            'date': str(datetime.now()),
                            'league': "PopUp_Event", # Just for the history log
                            't1_p1': m['t1'][0], 't1_p2': m['t1'][1],
                            't2_p1': m['t2'][0], 't2_p2': m['t2'][1],
                            'score_t1': s1, 'score_t2': s2,
                            'match_type': popup_name
                        }
                        
                        # --- THE MAGIC LINE: CALL THE NEW OVERALL-ONLY FUNCTION ---
                        process_overall_only_match(match_data)
                        
                        new_matches.append(match_data)
                
                # Save Log
                if new_matches:
                    new_df = pd.DataFrame(new_matches)
                    df_matches = pd.concat([df_matches, new_df], ignore_index=True)
                    ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
                    
                    st.success(f"✅ Processed {len(new_matches)} matches! Overall ratings updated.")
                    st.rerun()
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

# --- TAB 5: MATCH LOG & LADDER TOOLS ---
with tab5:
    st.header("Admin Tools")
    
    # --- SECTION A: LADDER UPLOAD (NEW ISLAND SYSTEM) ---
    st.subheader("📤 Upload Ladder Matches (Island System)")
    st.info("Upload a CSV with columns: 'winner', 'loser'. Optional: 'ladder'")
    
    ladder_upload = st.file_uploader("Upload CSV", type=["csv"], key="ladder_up")
    
    if ladder_upload is not None:
        target_ladder = st.selectbox("Select Ladder Name (if not in CSV)", 
                                     ["Testing_Ladder", "1v1", "Doubles", "Sniper"])
        
        if st.button("🚀 Process Ladder Matches"):
            df_ladder = pd.read_csv(ladder_upload)
            logs = process_batch_upload(df_ladder, ladder_name_from_ui=target_ladder)
            
            if logs:
                st.success(f"Processed {len(logs)} matches!")
                with st.expander("View Log"):
                    st.write(logs)
            else:
                st.warning("No matches processed. Check CSV headers.")

    st.divider()
    
    # --- SECTION B: RESTORE LOST LADDERS ---
    st.subheader("🔄 Restore Lost Ladders")
    st.info("Use this to reconstruct leaderboards from your 'Matches' history.")
    
    if 'league' in df_matches.columns:
        historical_leagues = [x for x in df_matches['league'].unique() if x and str(x) != "nan"]
        
        if historical_leagues:
            c1, c2 = st.columns([2, 1])
            with c1:
                league_to_restore = st.selectbox("Select League to Restore", historical_leagues)
            with c2:
                st.write("") 
                st.write("") 
                if st.button("Reconstruct Island"):
                    with st.spinner(f"Replaying history for {league_to_restore}..."):
                        msg = replay_league_history(league_to_restore)
                    st.success(msg)
                    st.rerun()
        else:
            st.warning("No league history found in the 'Matches' sheet.")
    else:
        st.error("Your 'Matches' sheet is missing the 'league' column.")

    st.divider()

    # --- SECTION C: MATCH EDITOR (EXISTING SYSTEM) ---
    st.subheader("📝 Edit Main League Matches")
    st.info("Edits here update the 'Matches' sheet directly.")
    
    edited_df = st.data_editor(df_matches, num_rows="dynamic", use_container_width=True)
    
    if st.button("💾 Save Changes & Recalc All History"):
        df_matches = edited_df
        df_players, df_matches = recalculate_all_stats(df_players, df_matches)
        ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
        ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
        st.success("Cloud Updated!")


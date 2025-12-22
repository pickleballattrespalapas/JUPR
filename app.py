import streamlit as st
import pandas as pd
import gspread
import math
import time
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import io

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

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
        # Fetch ALL worksheets in one meta-call if possible, 
        # but at least fetch all data from each sheet in one go.
        players_ws = sh.worksheet("Players")
        matches_ws = sh.worksheet("Matches")
        ratings_ws = sh.worksheet("player_ratings")
        
        # get_all_records() is one API call per sheet.
        df_players = pd.DataFrame(players_ws.get_all_records())
        df_ratings = pd.DataFrame(ratings_ws.get_all_records())
        df_matches = pd.DataFrame(matches_ws.get_all_records())

        # Safety checks for empty dataframes
        if df_ratings.empty:
            df_ratings = pd.DataFrame(columns=['name', 'ladder_id', 'rating'])
        
        # Numeric cleanup
        for c in ['score_t1', 'score_t2']:
            if c in df_matches.columns:
                df_matches[c] = pd.to_numeric(df_matches[c], errors='coerce').fillna(0)
            
        return df_players, df_ratings, df_matches, players_ws, matches_ws, ratings_ws
    except Exception as e:
        st.error(f"Database Quota Error: {e}. Please wait 60 seconds and refresh.")
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

def get_effective_rating(all_ratings, player_name, ladder_id):
    """
    Finds a player's rating for a specific island. 
    If not found, it pulls their 'starting_elo' from the master registry.
    """
    # 1. Search the Island Ratings first
    for r in all_ratings:
        if str(r['name']).strip() == str(player_name).strip() and str(r['ladder_id']).strip() == str(ladder_id).strip():
            return float(r['rating'])

    # 2. SOFT SEEDING: Look in the master df_players if Island rating is missing
    # This uses the 173 players from your Tres Palapas DB
    master_player = df_players[df_players['name'].str.strip() == str(player_name).strip()]
    
    if not master_player.empty:
        # Pull the starting_elo we restored earlier
        return float(master_player.iloc[0]['starting_elo'])

    # 3. FINAL FALLBACK: Only if player isn't in the registry at all
    return 1400.0 # Standard 3.5 JUPR starting point

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
    sh = get_db_connection()
    ratings_ws = sh.worksheet("player_ratings")
    matches_ws = sh.worksheet("Matches")
    
    # FETCH ONCE
    all_rows = ratings_ws.get_all_records()
    all_matches = matches_ws.get_all_records()
    df_history = pd.DataFrame(all_matches)
    
    league_matches = df_history[df_history['league'] == target_league]
    if league_matches.empty:
        return f"❌ No matches found for '{target_league}'."

    count = 0
    # Process everything in the 'all_rows' list (memory), NOT the API
    for _, row in league_matches.iterrows():
        s1, s2 = row.get('score_t1', 0), row.get('score_t2', 0)
        p1, p2 = row.get('t1_p1'), row.get('t1_p2')
        p3, p4 = row.get('t2_p1'), row.get('t2_p2')
        
        # Determine Winners/Losers
        if s1 > s2:
            win_team, lose_team = [p1, p2], [p3, p4]
        else:
            win_team, lose_team = [p3, p4], [p1, p2]

        # Update Memory for both League and Overall
        for context in [target_league, "OVERALL"]:
            # Helper logic done in memory
            w_ratings = [get_effective_rating(all_rows, p, context) for p in win_team if p]
            l_ratings = [get_effective_rating(all_rows, p, context) for p in lose_team if p]
            
            if len(w_ratings) > 0 and len(l_ratings) > 0:
                w_avg = sum(w_ratings) / len(w_ratings)
                l_avg = sum(l_ratings) / len(l_ratings)
                exp = 1 / (1 + 10 ** ((l_avg - w_avg) / 400))
                delta = K_FACTOR * (1 - exp)
                
                for p in win_team:
                    if p: update_local_memory(all_rows, p, context, get_effective_rating(all_rows, p, context) + delta)
                for p in lose_team:
                    if p: update_local_memory(all_rows, p, context, get_effective_rating(all_rows, p, context) - delta)
        count += 1

    # UPLOAD ONCE
    headers = list(all_rows[0].keys())
    data_to_write = [headers] + [list(r.values()) for r in all_rows]
    ratings_ws.clear()
    ratings_ws.update(data_to_write)
    
    return f"✅ Success! Replayed {count} matches without hitting API limits."

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
def get_match_schedule(format_type, players):
    # Ensure we have enough players
    if len(players) < int(format_type.split('-')[0]):
        return []

    # --- 12-PLAYER INDIVIDUAL RR SCHEDULE (11 ROUNDS) ---
    if format_type == "12-Player":
        # Standard pairings (1-based indices converted to 0-based for Python)
        raw_schedule = [
            [([2, 5], [3, 10]), ([4, 6], [8, 9]), ([11, 0], [1, 7])],  # Round 1 (3/6 v 4/11, 5/7 v 9/10, 12/1 v 2/8)
            [([5, 8], [6, 2]), ([7, 9], [0, 1]), ([11, 3], [4, 10])], # Round 2
            [([10, 1], [3, 4]), ([11, 6], [7, 2]), ([8, 0], [9, 5])], # Round 3
            [([11, 9], [10, 5]), ([0, 3], [1, 8]), ([2, 4], [6, 7])], # Round 4
            [([3, 6], [4, 0]), ([5, 7], [9, 10]), ([11, 1], [2, 8])], # Round 5
            [([8, 10], [1, 2]), ([11, 4], [5, 0]), ([6, 9], [7, 3])], # Round 6
            [([11, 7], [8, 3]), ([9, 1], [10, 6]), ([0, 2], [4, 5])], # Round 7
            [([1, 4], [2, 9]), ([3, 5], [7, 8]), ([11, 10], [0, 6])], # Round 8
            [([6, 8], [10, 0]), ([4, 7], [5, 1]), ([11, 2], [3, 9])], # Round 9
            [([11, 5], [6, 1]), ([9, 0], [2, 3]), ([7, 10], [8, 4])], # Round 10
            [([10, 2], [0, 7]), ([11, 8], [9, 4]), ([1, 3], [5, 6])], # Round 11
        ]
        
        matches = []
        for r_idx, round_pairs in enumerate(raw_schedule):
            for m_idx, (t1_idx, t2_idx) in enumerate(round_pairs):
                matches.append({
                    'desc': f"R{r_idx+1} C{m_idx+1}",
                    't1': [players[t1_idx[0]], players[t1_idx[1]]],
                    't2': [players[t2_idx[0]], players[t2_idx[1]]]
                })
        return matches

    # ... (Keep your existing 4-Player, 5-Player, 8-Player logic here) ...
    return []
    
    if format_type == "4-Player":
        matches = [{'t1':[p[0],p[1]],'t2':[p[2],p[3]],'desc':'R1'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R2'}, {'t1':[p[0],p[3]],'t2':[p[1],p[2]],'desc':'R3'}]
    elif format_type == "5-Player":
        matches = [{'t1':[p[1],p[4]],'t2':[p[2],p[3]],'desc':'R1 (1 Sit)'}, {'t1':[p[0],p[4]],'t2':[p[1],p[2]],'desc':'R2 (4 Sit)'}, {'t1':[p[0],p[3]],'t2':[p[2],p[4]],'desc':'R3 (2 Sit)'}, {'t1':[p[0],p[1]],'t2':[p[3],p[4]],'desc':'R4 (3 Sit)'}, {'t1':[p[0],p[2]],'t2':[p[1],p[3]],'desc':'R5 (5 Sit)'}]
    elif format_type == "6-Player":
        matches = [{'t1':[p[0],p[1]],'t2':[p[2],p[4]],'desc':'R1 (4,6 Sit)'}, {'t1':[p[2],p[5]],'t2':[p[0],p[4]],'desc':'R2 (1,2 Sit)'}, {'t1':[p[1],p[3]],'t2':[p[4],p[5]],'desc':'R3 (1,3 Sit)'}, {'t1':[p[0],p[5]],'t2':[p[1],p[2]],'desc':'R4 (3,4 Sit)'}, {'t1':[p[0],p[3]],'t2':[p[1],p[4]],'desc':'R5 (2,5 Sit)'}]
    elif format_type == "8-Player":
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

# --- TABS DEFINITION ---
# Public tabs: Leaderboards, Player Search. Admin tabs: 2-5.
tab_titles = [
    "🏆 Leaderboards", 
    "🔍 Player Search", 
    "🏟️ Live Court Manager (Admin)", 
    "🔄 Pop-Up RR (Admin)", 
    "👥 Players (Admin)", 
    "📝 Match Log (Admin)"
]
tab1, tab_search, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

# --- TAB 1: LEADERBOARDS (PUBLIC) ---
with tab1:
    with st.expander("ℹ️ How the Individual League Rating System Works"):
        st.markdown("""
        ### Why do I have multiple ratings?
        We use an Individual League Rating System to ensure fair play across different groups while maintaining a global skill level.
        
        * **Specific Ladders:** Your rating here is unique to this specific group. This protects the ladder's integrity; matches played elsewhere won't affect your standing here.
        * **OVERALL Rating (The Global Map):** Every match you play updates your Overall Rating. When you join a new league, this is the rating used for your initial seeding.
        """)
        
    st.divider()
    col_a, col_b = st.columns([1, 3])
    
    available_ladders = ["OVERALL"]
    if not df_ratings.empty:
        others = [str(x).strip() for x in df_ratings['ladder_id'].unique() if str(x).strip() != "OVERALL"]
        available_ladders += sorted(list(set(others)))
    
    query_params = st.query_params
    default_index = 0
    if "league" in query_params:
        target_from_link = query_params["league"]
        if target_from_link in available_ladders:
            default_index = available_ladders.index(target_from_link)

    with col_a:
        st.subheader("Filter")
        selected_ladder = st.selectbox("Select Ladder", available_ladders, index=default_index)
        if selected_ladder != "OVERALL":
            st.query_params["league"] = selected_ladder
        else:
            if "league" in st.query_params: del st.query_params["league"]

    with col_b:
        st.subheader(f"Standings: {selected_ladder}")
        if selected_ladder != "OVERALL":
            base_url = "https://8lkemld946rmtwwptk2gcs.streamlit.app/" 
            share_link = f"{base_url}?league={selected_ladder.replace(' ', '%20')}"
            st.markdown(f"**🔗 Share this leaderboard:**")
            st.code(share_link, language="text")
    
    if not df_ratings.empty:
        ladder_ratings = df_ratings[df_ratings['ladder_id'].str.strip() == selected_ladder].copy()
        relevant_matches = df_matches if selected_ladder == "OVERALL" else df_matches[df_matches['league'].str.strip() == selected_ladder]

        stats = {}
        for _, row in relevant_matches.iterrows():
            s1, s2 = row.get('score_t1', 0), row.get('score_t2', 0)
            p1, p2, p3, p4 = row.get('t1_p1'), row.get('t1_p2'), row.get('t2_p1'), row.get('t2_p2')
            wins, losers = ([p1, p2], [p3, p4]) if s1 > s2 else ([p3, p4], [p1, p2])
            for p in wins:
                if p:
                    if p not in stats: stats[p] = {'w': 0, 'l': 0}
                    stats[p]['w'] += 1
            for p in losers:
                if p:
                    if p not in stats: stats[p] = {'w': 0, 'l': 0}
                    stats[p]['l'] += 1

        ladder_ratings['wins'] = ladder_ratings['name'].map(lambda x: stats.get(x, {'w':0})['w'])
        ladder_ratings['losses'] = ladder_ratings['name'].map(lambda x: stats.get(x, {'l':0})['l'])
        ladder_ratings['matches'] = ladder_ratings['wins'] + ladder_ratings['losses']
        ladder_ratings['JUPR'] = (ladder_ratings['rating'] / 400).map('{:,.3f}'.format)
        ladder_ratings['Win %'] = (ladder_ratings['wins'] / ladder_ratings['matches'] * 100).fillna(0).map('{:.1f}%'.format)
        leaderboard = ladder_ratings[ladder_ratings['matches'] > 0].sort_values(by='rating', ascending=False)
        st.dataframe(leaderboard[['name', 'JUPR', 'matches', 'wins', 'losses', 'Win %']], use_container_width=True, hide_index=True)

# --- TAB: PLAYER SEARCH (PUBLIC) ---
with tab_search:
    st.header("🔍 Player Profile Search")
    
    # Improved Search: Starts blank
    all_player_names = [""] + sorted(df_players['name'].unique().tolist())
    selected_p = st.selectbox("Type to Search for a Player", all_player_names, index=0)
    
    if selected_p != "":
        st.subheader(f"Current Ratings: {selected_p}")
        p_ratings = df_ratings[df_ratings['name'] == selected_p].copy()
        if not p_ratings.empty:
            p_ratings['JUPR_val'] = (p_ratings['rating'] / 400).map('{:,.3f}'.format)
            cols = st.columns(len(p_ratings))
            for i, (_, row) in enumerate(p_ratings.iterrows()):
                cols[i].metric(label=row['ladder_id'], value=row['JUPR_val'])
        
        st.divider()
        
        st.subheader("Match History & Rating Changes")
        p_matches = df_matches[(df_matches['t1_p1'] == selected_p) | (df_matches['t1_p2'] == selected_p) | 
                               (df_matches['t2_p1'] == selected_p) | (df_matches['t2_p2'] == selected_p)].copy()
        
        if not p_matches.empty:
            def get_match_summary(row):
                is_t1 = (row['t1_p1'] == selected_p or row['t1_p2'] == selected_p)
                score_us, score_them = (row['score_t1'], row['score_t2']) if is_t1 else (row['score_t2'], row['score_t1'])
                raw_delta = row.get('elo_change_t1' if is_t1 else 'elo_change_t2', 0)
                jupr_delta = raw_delta / 400
                res = "✅ Win" if score_us > score_them else "❌ Loss"
                return pd.Series([res, f"{score_us}-{score_them}", round(jupr_delta, 3)])

            p_matches[['Result', 'Score', 'Δ JUPR']] = p_matches.apply(get_match_summary, axis=1)
            
            # Display Table (Sorting newest first)
            p_matches['date'] = pd.to_datetime(p_matches['date'])
            st.dataframe(
                p_matches.sort_values('date', ascending=False)[['date', 'league', 'Result', 'Score', 'Δ JUPR', 't1_p1', 't1_p2', 't2_p1', 't2_p2']], 
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No match history found for this player.")
    else:
        st.info("Please search and select a player to view their profile.")

# --- ADMIN PROTECTION ---
if not st.session_state.admin_logged_in:
    # If not logged in, show the login form in the restricted tabs
    for admin_tab in [tab2, tab3, tab4, tab5]:
        with admin_tab:
            st.warning("🔒 Admin Access Required")
            with st.form(key=f"login_gate_{admin_tab}"):
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Unlock All Admin Tabs"):
                    if pwd == st.secrets["admin_password"]:
                        st.session_state.admin_logged_in = True
                        st.success("Access Granted! Tabs Unlocked.")
                        st.rerun()
                    else:
                        st.error("Incorrect Password")
else:
    # --- EVERYTHING BELOW THIS IS ONLY VISIBLE IF LOGGED IN ---
    
    # 1. Sidebar Logout
    if st.sidebar.button("🔒 Logout Admin"):
        st.session_state.admin_logged_in = False
        st.rerun()

    # 2. TAB 2: LIVE COURT MANAGER
    with tab2:
        st.header("Live Court Manager") 
        with st.expander("Setup", expanded=True):
            event_date = st.date_input("Match Date", datetime.now(), key="date_tab2")
            league_name = st.text_input("League", "Fall 2025 Ladder")
            num_courts = st.number_input("Courts", 1, 20, 1)
            
            with st.form("setup"):
                court_data = []
                for i in range(num_courts):
                    c1, c2 = st.columns([1,4])
                    with c1: 
                        t = st.selectbox(f"Type {i+1}", ["4-Player","5-Player","6-Player","8-Player", "12-Player"], key=f"t{i}")
                    with c2: 
                        n = st.text_area(f"Names {i+1}", key=f"n{i}", height=68)
                    court_data.append({'id':i+1, 'type':t, 'names':n})
                
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
                        c1, c2, c3, c4 = st.columns([3,1,1,3])
                        with c1: st.text(f"{m['desc']} | {m['t1'][0]} & {m['t1'][1]}")
                        with c2: s1 = st.number_input("S1", 0, key=f"s_{c['court']}_{i}_1")
                        with c3: s2 = st.number_input("S2", 0, key=f"s_{c['court']}_{i}_2")
                        with c4: st.text(f"{m['t2'][0]} & {m['t2'][1]}")
                
                if st.form_submit_button("Submit & Save to Cloud"):
                    new_matches = []
                    for c in st.session_state.schedule:
                        for i, m in enumerate(c['matches']):
                            s1, s2 = st.session_state.get(f"s_{c['court']}_{i}_1", 0), st.session_state.get(f"s_{c['court']}_{i}_2", 0)
                            if s1 == 0 and s2 == 0: continue
                            
                            match_data = {
                                'id': len(df_matches) + len(new_matches) + 1, 
                                'date': str(event_date), 
                                'league': league_name, 
                                't1_p1': m['t1'][0], 't1_p2': m['t1'][1], 
                                't2_p1': m['t2'][0], 't2_p2': m['t2'][1], 
                                'score_t1': s1, 'score_t2': s2, 
                                'match_type': f"Court {c['court']} RR"
                            }
                            process_live_doubles_match(match_data, ladder_name=league_name)
                            new_matches.append(match_data)
                    
                    if new_matches:
                        new_df = pd.DataFrame(new_matches)
                        df_matches = pd.concat([df_matches, new_df], ignore_index=True)
                        ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
                        st.success(f"✅ Processed {len(new_matches)} matches!")
                        st.rerun()

    # 3. TAB 3: POP-UP ROUND ROBIN
    with tab3:
        st.header("Pop-Up Round Robin")
        with st.expander("Event Setup", expanded=True):
            event_date_rr = st.date_input("Event Date", datetime.now(), key="date_tab3")
            popup_name = st.text_input("Event Name", f"PopUp {datetime.now().strftime('%Y-%m-%d')}")
            rr_courts = st.number_input("Number of Courts", 1, 20, 1, key="rr_courts")
            
            with st.form("rr_setup"):
                rr_data = []
                for i in range(rr_courts):
                    c1, c2 = st.columns([1, 4])
                    with c1: 
                        t = st.selectbox(f"Format {i+1}", ["4-Player", "5-Player", "6-Player", "8-Player", "12-Player"], key=f"rr_t{i}")
                    with c2: 
                        n = st.text_area(f"Names {i+1}", key=f"rr_n{i}", height=68, placeholder="Joe, Kevin, Scott, Robin...")
                    rr_data.append({'id': i+1, 'type': t, 'names': n})
                
                if st.form_submit_button("Generate Schedule"):
                    st.session_state.rr_schedule = []
                    for c in rr_data:
                        pl = [x.strip() for x in c['names'].replace('\n', ',').split(',') if x.strip()]
                        st.session_state.rr_schedule.append({'court': c['id'], 'matches': get_match_schedule(c['type'], pl)})

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
                    for c in st.session_state.rr_schedule:
                        for i, m in enumerate(c['matches']):
                            s1, s2 = st.session_state.get(f"rr_s_{c['court']}_{i}_1", 0), st.session_state.get(f"rr_s_{c['court']}_{i}_2", 0)
                            if s1 == 0 and s2 == 0: continue
                            match_data = {'date': str(event_date_rr), 'league': "PopUp_Event", 't1_p1': m['t1'][0], 't1_p2': m['t1'][1], 't2_p1': m['t2'][0], 't2_p2': m['t2'][1], 'score_t1': s1, 'score_t2': s2}
                            process_overall_only_match(match_data)
                    st.success("✅ Overall ratings updated!")
                    st.rerun()

    # 4. TAB 4: PLAYER MANAGEMENT
    with tab4:
        st.header("Player Management")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("➕ Add New Player")
            with st.form("add_p"):
                n = st.text_input("Name")
                r = st.number_input("Start JUPR", 3.0, step=0.1)
                if st.form_submit_button("Add Player"):
                    if n and n not in df_players['name'].values:
                        new_p = {'name': n, 'elo': r*400, 'starting_elo': r*400, 'matches_played':0, 'wins':0, 'losses':0}
                        df_players = pd.concat([df_players, pd.DataFrame([new_p])], ignore_index=True)
                        ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
                        st.success(f"Added {n} to Cloud")
                        st.rerun()
                    else:
                        st.error("Invalid name or player already exists.")
        with c2:
            st.subheader("🗑️ Delete Player")
            p_to_delete = st.selectbox("Select Player to Remove", [""] + sorted(df_players['name'].tolist()))
            if p_to_delete:
                st.warning(f"Are you sure you want to delete **{p_to_delete}**?")
                if st.button(f"Confirm Delete {p_to_delete}"):
                    df_players = df_players[df_players['name'] != p_to_delete]
                    ws_players.clear()
                    ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
                    if not df_ratings.empty:
                        df_ratings = df_ratings[df_ratings['name'] != p_to_delete]
                        ws_ratings.clear()
                        ws_ratings.update([df_ratings.columns.values.tolist()] + df_ratings.values.tolist())
                    st.success(f"Successfully deleted {p_to_delete}.")
                    st.rerun()
        st.divider()
        st.subheader("Current Player Registry")
        st.dataframe(df_players, use_container_width=True, hide_index=True)

    # 5. TAB 5: ADMIN TOOLS
    with tab5:
        st.header("Admin Tools")
        st.subheader("📤 Upload Ladder Matches")
        ladder_upload = st.file_uploader("Upload CSV", type=["csv"], key="ladder_up")
        if ladder_upload is not None:
            target_ladder = st.selectbox("Select Ladder Name", ["Testing_Ladder", "1v1", "Doubles", "Sniper"])
            if st.button("🚀 Process Ladder Matches"):
                df_ladder = pd.read_csv(ladder_upload)
                logs = process_batch_upload(df_ladder, ladder_name_from_ui=target_ladder)
                st.write(logs)

        st.divider()
        st.subheader("🔄 Reconstruct/Reset League")
        if 'league' in df_matches.columns:
            hist_leagues = [x for x in df_matches['league'].unique() if x and str(x) != "nan"]
            league_to_restore = st.selectbox("Select League to Fix", hist_leagues)
            if st.button("Clean Reconstruct"):
                with st.spinner(f"Wiping old data and replaying history for {league_to_restore}..."):
                    sh = get_db_connection()
                    r_ws = sh.worksheet("player_ratings")
                    all_r = r_ws.get_all_records()
                    clean_r = [r for r in all_r if r['ladder_id'] != league_to_restore and r['ladder_id'] != 'OVERALL']
                    if clean_r:
                        headers = list(all_r[0].keys())
                        data_to_write = [headers] + [list(r.values()) for r in clean_r]
                        r_ws.clear()
                        r_ws.update(data_to_write)
                    else:
                        r_ws.clear()
                    msg = replay_league_history(league_to_restore)
                    st.success(f"Fixed! {msg}")
                    st.rerun()

        st.divider()
        st.subheader("📝 Edit Match History")
        edited_df = st.data_editor(df_matches, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save & Recalc"):
            df_matches = edited_df
            df_players, df_matches = recalculate_all_stats(df_players, df_matches)
            ws_players.update([df_players.columns.values.tolist()] + df_players.values.tolist())
            ws_matches.update([df_matches.columns.values.tolist()] + df_matches.values.tolist())
            st.success("Cloud Updated!")

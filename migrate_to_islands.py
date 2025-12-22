import gspread

# 1. Connect to Google Sheets
gc = gspread.service_account(filename='credentials.json') # Use your actual json file name
sh = gc.open("Your Spreadsheet Name") # Use your actual sheet name

# 2. Open the worksheets
users_sheet = sh.worksheet("Users") # Or whatever your main sheet is called
ratings_sheet = sh.worksheet("player_ratings")

# 3. Get all current user data
# Assuming your Users sheet has columns like: [ID, Name, Current_Rating, ...]
all_users = users_sheet.get_all_records()

print(f"Found {len(all_users)} users. Migrating...")

# 4. Prepare the new rows
new_rows = []

for user in all_users:
    # Adjust these keys based on your actual column headers in the Users sheet
    u_id = user['Username']  # or user['id']
    current_rating = user['Rating'] # or user['Elo']
    
    # Create the "OVERALL" entry
    new_rows.append([u_id, 'OVERALL', current_rating])

# 5. Write to the new tab
if new_rows:
    ratings_sheet.append_rows(new_rows)
    print("Success! Data migrated to 'player_ratings' tab.")
else:
    print("No users found to migrate.")
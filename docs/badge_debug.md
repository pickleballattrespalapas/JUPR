# Badge Debug (Phase 0)

## Enable the view

1. Log in as an admin in the Streamlit app.
2. Set one of the following environment variables before starting the app:
   - `BADGE_DEBUG=1`
   - `JUPR_ADMIN=1`

With admin login + one of the flags enabled, the **🧪 Badge Debug** page appears in the admin sidebar. If the flag is missing, the page shows a friendly “disabled” message and exits early.

## Use it to debug a player (e.g., Tyson)

1. Open **🧪 Badge Debug**.
2. Select:
   - Club (defaults to the active club)
   - League (or “All leagues”)
   - Player (searchable by name/id)
   - Badge
3. Click **Run Badge Debug**.

The page renders a single “truth table” row with counts, the evaluator candidate rows (including `context_id`, `match_id`, and `value_json`), and a full match filter audit that lists which match IDs were removed at each step. Use the raw vs filtered match ID expanders to pinpoint mismatches quickly.

# Live Brackets MVP

A browser-only prototype for running:
- round robins
- ladder / league nights
- tournaments

## Files
- `index.html` — app shell
- `styles.css` — UI styling
- `engine.js` — scheduling and standings engine
- `app.js` — browser UI
- `demo-round-robin.json` — sample import
- `demo-league.json` — sample import
- `demo-tournament.json` — sample import

## How to use
1. Keep all files in the same folder.
2. Open `index.html` in a browser.
3. Create an event or import one of the demo JSON files.
4. Tournament setup expects one team per line.
5. Scores autosave in local storage.
5. Export JSON whenever you want a backup.

## Current capabilities
- Standard round robin for any count
- Switch-partner doubles RR for 4, 5, 8, and 12 players
- Live ladder with court-by-court standings and editable movement
- Single-elimination team tournament bracket
- Keyboard-first score entry with tab / Enter progression
- Print-friendly screens
- No backend required

## Notes
This MVP was designed from:
- the uploaded RR chart CSVs
- the uploaded JUPR pages for match upload, Moneyball, and league management

## Next step
The clean migration path is:
1. put this folder into a GitHub repo
2. keep `engine.js` as the source-of-truth logic layer
3. later wrap the UI in React/Vite and add hosted accounts

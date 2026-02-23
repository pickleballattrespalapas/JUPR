# Session Ladder Engine MVP — Manager Guide

This guide explains how to run a Session Ladder from setup to publish.

## Feature flags
- `FEATURE_SESSION_LADDER`: enables Session Ladder UI/API.
- `FEATURE_LIVE_SCORING`: reserved for future live-scoring UX.
  - The MVP already stores each game as first-class records (`session_ladder_games`), so live scoring can be layered in without changing core data model.

## Step 1: Roster intake
The Session Console supports four intake modes:
1. **Search/select existing player**
   - Pick an existing player and add to roster with status + rating snapshot.
2. **Manual new player**
   - Create a minimal player profile and immediately add to roster.
3. **Copy/paste names**
   - Paste newline/comma-separated names.
   - Fuzzy preview marks each as exact/fuzzy/new before confirm.
4. **CSV upload**
   - Upload CSV, map columns, preview normalized rows, confirm add.

### Roster statuses
- `EXPECTED`
- `CHECKED_IN`
- `NO_SHOW`
- `WALK_IN`

### Duplicate handling
- Intake normalizes and de-duplicates names where possible.
- Session roster uses a unique `(session_id, player_id)` constraint to prevent duplicates.

## Step 2: Seeding rules
- Seeding sorts eligible players by `rating_snapshot` (descending).
- Uses uniform court sizes (`players_per_court`: 4 or 5).
- Creates as many full courts as possible.
- Extra players beyond court capacity are effectively **standby** (not assigned to a court pod).

## Step 3: Scoring flow
- Enter scores game-by-game on each court sheet.
- For each row: select winner + losing points (0–10).
- Winner score is auto-set to 11 (league default in MVP UI flow).
- Autosave writes each game immediately.
- Court standings and player stats update from recorded games:
  - W/L
  - PF
  - PA
  - PD

## Step 4: Tie-break rules
Tie resolution order is fixed:
1. Wins
2. Point Differential (PD)
3. Points For (PF)
4. Head-to-head (games where tied players were opponents)
5. `PlayoffRequired`

If unresolved at step 5, manager can record playoff winner in the workflow.

## Step 5: Movement rules
- Movement distance is always **exactly 1 court**.
- Configurable `movers_per_court` is `1` or `2`.
- Top movers move up one court; bottom movers move down one court.
- Boundary courts are handled automatically.

## Step 6: Completion + publish
- If all required scores are entered for the final planned round, the session can complete immediately at closeout.
- **Complete Session**:
  - Runs rating update hook through existing rating logic.
  - Idempotent (safe to run again; no double-apply).
  - Writes session-linked rating history rows.
- **Publish Session**:
  - Locks results (post-complete/publish edits blocked).
  - Stores session recap snapshot.
  - Stores ratings leaderboard snapshot.
  - Applies attendance increments.

## Awards eligibility
- Attendance accumulates per player for awards eligibility.
- Minimum sessions threshold is configurable via:
  - `SESSION_LADDER_MIN_AWARD_SESSIONS` (default: 4)

## Dashboard signals (League Manager)
Session Ladder dashboard summary includes:
- players in latest session
- current step/state
- ratings snapshot size
- standings visibility for latest round
- awards-eligible count

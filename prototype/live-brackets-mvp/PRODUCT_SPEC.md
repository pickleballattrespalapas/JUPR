# Live Brackets MVP — Product Spec

## Product position
This is the browser-first redesign of the printable round-robin / bracket workflow:
a live event runner that makes it easy for any organizer to create an event, enter scores as matches happen, and immediately see standings, winners, and movement between rounds.

The MVP is intentionally local-first:
- no database
- no ratings
- no account requirement
- no backend dependency

Everything runs in the browser and can be backed up as JSON.

## Primary user flow
1. Choose event type:
   - Round Robin
   - League / Ladder
   - Tournament
2. Enter event name.
3. Choose player count.
4. Enter player names.
5. Start the event.
6. Enter scores live as matches finish.
7. See automatic standings / bracket advancement / movement suggestions.
8. Export the event JSON or print the view.

## Event types in this build

### 1) Round Robin
Two modes:
- Standard round robin for any participant count.
- Switch-partner doubles using the uploaded 4, 5, 8, and 12-player chart patterns.

Computed live:
- wins
- losses
- ties
- games played
- points for
- points against
- differential
- live leader / winner

### 2) League / Ladder
Built for club-night style events where players are split across courts.

Current MVP rules:
- uses 4-player and 5-player courts
- generates switch-partner mini-rounds inside each court
- computes per-court standings
- suggests one-up / one-down movement
- allows manual next-round overrides before the next round starts
- preserves completed round history

Computed live:
- current-court standings
- cumulative ladder standings
- proposed court movement
- final event completion status

### 3) Tournament
Single-elimination bracket with:
- seeded order
- automatic byes
- automatic advancement when winners are known
- champion display

## Why this structure
The uploaded RR charts define the actual schedule patterns for switch-partner doubles, while the uploaded JUPR pages show three clear product behaviors:
- fast score entry
- live standings
- live ladder management with between-round changes

This MVP turns those ideas into one browser-only product foundation.

## Technical design
- Static HTML/CSS/JS
- No framework required
- LocalStorage autosave
- JSON import/export
- Print-friendly layout
- Engine logic separated from UI (`engine.js`)

## Data model
Each event serializes to plain JSON and contains:
- event metadata
- participants
- rounds
- matches
- scores
- movement assignments when relevant

## MVP constraints
- Switch-partner doubles templates: 4, 5, 8, 12
- League court sizes: 4 or 5 only
- Tournament format: single elimination only

## Natural phase 2
- hosted persistence
- club accounts
- public read-only share links
- director / scorer roles
- saved club templates
- multi-stage tournaments
- playoffs / consolation brackets
- player availability and substitutes
- analytics and event archive

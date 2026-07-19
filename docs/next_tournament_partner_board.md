# Next Tournament Partner Board

This document tracks the public Tournament Partner Board migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- Next route: `/clubs/[clubSlug]/tournament-partner-board`.
- Uses the public Tournament Roster FastAPI projection: `GET /clubs/{club_slug}/tournament-roster`.
- Query support matching registration pages: `registration_slug` / `tournament` and `tournament_id`.
- Read-only list of players who registered as needing a partner.
- Public-safe event, division, skill, age-bracket, and note display.
- Opaque public board-entry references for anchors and token-gated pairing interest.

## Data safety

The partner board reads only from the public roster projection. It intentionally does not expose private contact fields such as email or phone, exact age, DUPR IDs, or database row IDs. Direct pairing actions require a valid registration edit token; the browser sends an opaque board-entry key and FastAPI resolves it against the current tournament before writing.

No browser-side code reads Supabase directly. The browser calls FastAPI, and FastAPI delegates roster construction to the existing Python tournament registration domain code.

## Explicitly out of scope

- Organizer moderation controls.
- Email/contact disclosure.

Those remain follow-up slices so moderation, abuse prevention, and audit semantics can be reviewed independently.

## Follow-up slices

- Add organizer moderation and audit attribution.
- Add partner-request decline/cancel controls.

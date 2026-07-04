# Next Tournament Partner Board

This document tracks the public Tournament Partner Board migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- Next route: `/clubs/[clubSlug]/tournament-partner-board`.
- Uses the public Tournament Roster FastAPI projection: `GET /clubs/{club_slug}/tournament-roster`.
- Query support matching registration pages: `registration_slug` / `tournament` and `tournament_id`.
- Read-only list of players who registered as needing a partner.
- Public-safe event, division, skill, age-bracket, and note display.

## Data safety

The partner board reads only from the public roster projection. It intentionally does not expose private contact fields such as email or phone. It also does not expose direct request/accept actions in this slice.

No browser-side code reads Supabase directly. The browser calls FastAPI, and FastAPI delegates roster construction to the existing Python tournament registration domain code.

## Explicitly out of scope

- Public partner request creation.
- Partner request accept/decline actions.
- Organizer moderation controls.
- Email/contact disclosure.
- Tokenized edit links.

Those remain follow-up slices so moderation, abuse prevention, and audit semantics can be reviewed independently.

## Follow-up slices

- Add safe public partner request flow with anti-abuse controls.
- Add organizer moderation and audit attribution.
- Add tokenized registration edit route.

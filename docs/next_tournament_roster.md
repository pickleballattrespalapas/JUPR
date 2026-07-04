# Next Tournament Roster

This document tracks the public Tournament Roster migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI route: `GET /clubs/{club_slug}/tournament-roster`.
- Next route: `/clubs/[clubSlug]/tournament-roster`.
- Query support matching registration pages: `registration_slug` / `tournament` and `tournament_id`.
- Public-safe roster projection using the existing Python tournament registration roster compiler.
- Public-safe summary cards for registrations, players, waitlist, and players needing partners.
- Grouped roster entries by day/event/division.
- Public-safe “players looking for partners” visibility.
- Read-only partner-board page at `/clubs/[clubSlug]/tournament-partner-board`, backed by the same public-safe projection.

## Data safety

The route exposes only the public roster projection from Python. It intentionally omits private contact fields such as email and phone, internal tournament notes, registration builder drafts, and staff-only operational state.

No browser-side code reads Supabase directly. The browser calls FastAPI, and FastAPI delegates roster construction to the existing Python tournament registration domain code.

## Explicitly out of scope

- Partner-board request/accept actions.
- Public edit links.
- Admin registration management.
- Draw seeding, teams, games, podiums, and other tournament operations.

Those remain follow-up slices so moderation, abuse prevention, and audit semantics can be reviewed independently.

## Follow-up slices

- Add safe public partner request flow with anti-abuse controls.
- Add tokenized registration edit route.
- Add admin registration management after public intake and roster visibility are proven.

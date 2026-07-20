# Next Tournament Roster

This document tracks the public Tournament Roster migration from Streamlit to Next.js and FastAPI.

## Scope in this slice

- FastAPI route: `GET /clubs/{club_slug}/tournament-roster`.
- Next route: `/clubs/[clubSlug]/tournament-roster`.
- Query support matching registration pages: `registration_slug` / `tournament` and `tournament_id`.
- Public-safe roster projection using the existing Python tournament registration roster compiler.
- Public-safe summary cards for registrations, players, waitlist, and players needing partners.
- Grouped roster entries by day/event/division.
- Day, event, division, and status filters with shareable query strings and row anchors.
- Explicit `Registered`, `Waitlist`, `Needs Partner`, `Pending Partner Request`, and fail-closed `Review` labels.
- Registration open/close window display.
- Public-safe “players looking for partners” visibility.
- Public-safe skill and age-bracket metadata plus direct partner-board deep links.
- Read-only partner-board page at `/clubs/[clubSlug]/tournament-partner-board`, backed by the same public-safe projection.

## Data safety

The route exposes only the public roster projection from Python. It intentionally omits private contact fields such as email and phone, exact ages, DUPR IDs, database row IDs, internal tournament notes, registration builder drafts, and staff-only operational state. Free-text partner notes redact email addresses and phone-like values before they enter the public response.

Roster and partner-board anchors use stable opaque public references. They are locators, not credentials. Pairing writes still require a valid registration edit token, and FastAPI resolves the opaque target reference back to the current public board entry before allowing a request.

No browser-side code reads Supabase directly. The browser calls FastAPI, and FastAPI delegates roster construction to the existing Python tournament registration domain code.

## Explicitly out of scope

- Partner-board request/accept actions.
- Public edit links.
- Admin registration management.
- Draw seeding, teams, games, podiums, and other tournament operations.

Those remain follow-up slices so moderation, abuse prevention, and audit semantics can be reviewed independently.

## Automated closure evidence

- Status mapping tests cover confirmed, waitlist, needs-partner, pending, review, partner-missing, legacy unresolved, and unknown/null states.
- Recursive public-field denylist tests cover contact fields, exact age, DUPR ID, and registration/selection/player/source IDs.
- Public-service tests cover populated, empty, and unavailable/schema-error states.
- Staging browser smoke covers filtering and a shareable roster row deep link when the seeded roster is populated.

Manual responsive and representative-data acceptance remains required before changing the parity matrix row to `Done`.

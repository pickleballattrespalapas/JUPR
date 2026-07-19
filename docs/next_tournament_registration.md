# Next tournament registration parity

The public Next route is `/clubs/[clubSlug]/tournament-registration`. It uses
FastAPI for every read, preflight, and write. Browser code never initializes a
Supabase client and never receives a service-role credential.

## New-registration flow

The route now starts with an explicit choice:

1. **Start New** opens a four-step contact, profile, event/partner, and review
   wizard.
2. **Edit Existing** requests a non-enumerating secure edit link through the
   existing FastAPI recovery endpoint.

The first new-registration step calls
`POST /clubs/{club_slug}/tournament-registration/profile-resolution`. This
server-side preflight:

- requires first name, last name, email, age, and gender;
- checks the selected tournament and registration-open state;
- hands duplicate emails to the secure edit-link flow without returning a
  registration id;
- returns at most three exact, active, club-scoped profile suggestions; and
- excludes profile email, phone, and other private fields.

Profile suggestions are convenience hints, not identity proof. They can prefill
public rating and DUPR fields, but the submitted registration remains unlinked
(`player_id = null`) until tournament staff verify the relationship. This keeps
the security boundary introduced by the public registration service while
restoring the Streamlit profile-resolution experience.

## Selection and partner policy

FastAPI remains authoritative for event and partner validation. It rejects:

- duplicate event ids or more than one division in a day/family;
- closed, hidden, disabled, or moved event selections;
- partner data on singles events;
- partner-required events without `HAS_PARTNER` or `NEEDS_PARTNER`;
- self-partner email, invalid partner email, or a `HAS_PARTNER` entry missing
  partner name, age, or gender;
- gender/skill-ineligible player or partner combinations; and
- a public partner-board listing without explicit contact consent.

Free-text `HAS_PARTNER` details are explicitly presented as a staff-review
suggestion. They do not create a team. The dedicated partner request/accept
workflow remains separate and lands in the next stack slice.

## Duplicate, closed, and recovery behavior

- A duplicate new submission returns HTTP `409`; the browser switches to the
  secure edit-link flow instead of attempting a second write.
- A closed tournament disables **Start New** while leaving **Edit Existing**
  available.
- The profile preflight also reports a closed state before returning any profile
  candidates.
- The existing secure edit-link response remains non-enumerating.

## Public content and handoffs

The page renders configured sponsor, rules/registration notes, and refund-policy
content. It also provides a tournament-scoped link to the public roster before
the wizard. The successful-submit confirmation handoff is intentionally left on
the existing contract so the earlier confirmation-security stack slice can be
cherry-picked without duplicating or weakening it.

## Automated evidence

- `tests/test_public_tournament_registration_service.py` covers safe profile
  resolution, duplicate recovery, closed state, required demographics, singles,
  partner-required selections, and partner-board consent.
- `tests/test_api_contract_tournament_registration.py` covers the FastAPI
  profile-resolution contract, required demographics, and HTTP `409` recovery.
- `apps/web/e2e/tournament-registration.parity.spec.ts` covers the route-specific
  mode chooser, edit-link handoff, required fields, and browser-to-FastAPI
  profile preflight without writing staging data.

The route stays `Partial` until the final consolidated staging manual session.
Streamlit remains the operational fallback during that period.

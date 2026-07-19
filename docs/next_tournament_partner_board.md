# Next Tournament Partner Board

This is the automated-ready contract for the public Tournament Partner Board
migration. Streamlit remains available as the operator fallback until the final
manual parity session signs off this page.

## Functional flow

- Public route: `/clubs/[clubSlug]/tournament-partner-board`.
- Public reads come from `GET /clubs/{club_slug}/tournament-roster`, using the
  explicit `roster.partner_board_entries` projection. The broader
  `players_needing_partners` roster list is not treated as contact consent.
- A requester opens the board through their secure registration edit link and
  sends interest for one compatible selection in the same tournament/division.
- The requested player reviews incoming requests through their own edit token and
  may accept or decline. The requester may cancel an outgoing pending request.
- Acceptance creates one confirmed team, switches both selections to
  `HAS_PARTNER`, removes both from the public board, and cancels every competing
  pending request involving either selection.
- Query support matches registration pages: `registration_slug` / `tournament`
  and `tournament_id`.
- Public entries expose event, division, skill, age bracket, and an allowlisted
  note plus an opaque board reference; they never expose database row IDs.

All writes are FastAPI-mediated. The browser never initializes Supabase, never
receives the service-role key, and never sends email directly.

## Deterministic and stale-state behavior

The partner board intentionally omits email, phone, exact age, DUPR IDs, and
database row IDs. Direct pairing actions require a valid registration edit
token; the browser sends an opaque board-entry key and FastAPI resolves it
against the current tournament before writing.

`20260719194500_public_partner_pairing_lifecycle.sql` provides two service-role-only,
`SECURITY INVOKER` RPCs:

- `create_tournament_partner_request(...)` uses the foundation's universal
  registration/selection lock order, validates that both selections are still
  compatible, and returns the existing pending row for an exact retry. A partial
  unique index prevents concurrent duplicate requests.
- `transition_tournament_partner_request(...)` handles accept, decline, and cancel
  under the same lock protocol. Acceptance locks the complete competing-request
  graph before creating the team and cancelling conflicts. Repeating the same
  completed action succeeds idempotently and does not repeat notification
  delivery. A different action against terminal state returns an HTTP `409`
  stale-state response after actor ownership is checked.

The Python domain service is still the application authority: it verifies the
edit token/club/registration, selects the allowed transition, invokes the
transaction boundary, and builds only public-safe responses.

## Privacy and notification safety

- A public board entry requires the global and event partner-board switches,
  selection-level display consent (`show_on_partner_board`), and
  registration-level contact consent (`wants_partner_board_contact`). Needing a
  partner alone is not consent. Withdrawing consent or disabling the board makes
  a pending acceptance stale and cancels it without creating a team.
- Public request/review payloads are allowlisted and omit email, phone, partner
  contact, DUPR ID, notes, and edit tokens.
- Pairing emails contain names and a secure board action link, never requester or
  target contact details.
- `JUPR_TOURNAMENT_PARTNER_CONTACT_DENYLIST` accepts comma-separated exact email
  addresses or `@domain` entries. Denied recipients are skipped without exposing
  that address in a public response.
- Writes survive SMTP or configuration failure. The response reports only reduced
  delivery statuses (`dry_run`, `staging_redirect`, `sent`, `skipped`, `failed`,
  or `not_repeated`).
- Staging must use `JUPR_EMAIL_MODE=dry_run` or `staging_redirect`; unrestricted
  live delivery is not part of automated acceptance.

## Automated evidence

- `tests/test_public_tournament_partner_lifecycle_schema.py`: locks, unique pending
  pair, atomic team creation/cancellation, and service-role-only grants.
- `tests/test_public_tournament_partner_lifecycle.py`: exact retry, ownership,
  decline/cancel, accept, competing cancellation, stale state, and privacy.
- `tests/test_public_tournament_pairing_email_service.py` and
  `tests/test_tournament_pairing_lifecycle_email.py`: no-repeat delivery,
  write-survives-mail-failure, denylist behavior, and contact-safe copy.
- `tests/test_api_contract_tournament_partner_flow.py`: complete FastAPI lifecycle
  including idempotent retries and HTTP `409` stale behavior.
- `apps/web/e2e/tournament-partner-board.parity.spec.ts`: public privacy boundary
  plus a disposable create/review/cancel browser/API flow when the documented
  staging fixture variables are supplied.

## Staging fixture and rollback

The mutating browser check is opt-in and requires a disposable requester and board
target that are not imported into a draw:

```text
STAGING_PARTNER_TOURNAMENT_ID
STAGING_PARTNER_REGISTRATION_SLUG
STAGING_PARTNER_REQUESTER_EDIT_TOKEN
STAGING_PARTNER_REQUESTER_SELECTION_ID
STAGING_PARTNER_TARGET_BOARD_ENTRY_KEY
```

The automated browser flow always finishes by cancelling the request and verifies
the terminal `CANCELLED` state. The final manual session will separately exercise
acceptance on disposable registrations, inspect redirected email, verify competing
request cancellation, and restore the fixture through registration edit or the
Streamlit fallback.

The migration deliberately refuses to guess which row to keep if historical exact
duplicate pending requests exist. Its `JUPR_PARTNER_DUPLICATE_PENDING` preflight
must be resolved through the existing operator fallback before retrying; it never
silently deletes registration relationship data.

## Stack integration assumptions

This slice was developed from the parity foundation commit and is intended to sit
after the registration wizard PR. During stack integration:

1. retain the registration PR's required-field/profile/partner policy changes;
2. retain this slice's consent-filtered `partner_board_entries` projection;
3. retain the earlier stable edit-token secret/preflight behavior; and
4. apply the canonical Supabase lifecycle migration after the server-only Data API
   lockdown migration, then deploy FastAPI before Next.

No matrix row moves to `Done` in this PR. That happens only after the consolidated
manual acceptance session.

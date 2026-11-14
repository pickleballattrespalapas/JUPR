# Organizer-managed interclub leagues

The default `/admin/interclub` workspace manages a league in one organizer club account. Participating clubs are names, captain contacts and player rosters, not tenant accounts. The previous workflow remains at `/admin/interclub/legacy`.

The organizer uses Teams & Players, Schedule, Enter Results and Standings. Each generated meet can have a private host link. Hosts can view the schedule, print the existing paper packet, save scores and submit a completed meet without signing in. Only the organizer approves official results or changes the season.

## Records and publication

- `pcs_managed_interclub_seasons` belongs to the organizer's actual club. Participating club IDs inside `details` have no tenant/account relationship.
- `pcs_managed_interclub_meets` stores immutable roster and schedule snapshots once scoring begins, draft scores and a separate approved document.
- Approval updates league standings and the published league page. Draft scores never replace the last approved result. Reopening a meet requires a correction reason.
- The public league link is separate from the private host link and is suitable for Facebook announcements.
- This workflow calculates interclub standings and Club Cup points. It does not write individual JUPR rating effects, memberships or other clubs' league records. Existing rating records remain in the previous workflow.

## Host access and saves

Private links contain a 256-bit token in the URL fragment. The API receives it in POST bodies and stores only its SHA-256 digest. Host responses omit captain contacts and other meets. Link replacement and revocation are checked inside the same database transaction as score writes.

Every write checks the expected revision. The UI retains newer typing during in-flight saves, pauses on uncertain saves, and offers a download of unsaved scores. Conflicts lock further edits until the saved meet is reloaded. Hosts cannot publish, approve, reopen or edit the schedule. Automatic saving requires connectivity.

All three tables have RLS and no anonymous/authenticated grants. The service-only write function checks organizer assignments or the scoped host token and records an audit entry without token hashes. Staging mutations follow the existing write guards.

## Continuing an existing season

“Use simplified setup” copies an organizer-owned season's club labels, roster snapshots, dates and saved/approved score documents. Historical substitutes are resolved from the source season's entries and club-scoped profiles. It is idempotent, retains original records and does not migrate rating effects. Incompatible histories, such as multiple competition phases in a single meet, fail before writing and remain available in the previous workspace.

## Verification

```sh
python -m pytest -q tests/test_managed_interclub.py tests/test_interclub_competition.py tests/test_api_contract_interclub_competition.py tests/test_interclub_public_competition.py tests/test_interclub_setup.py
npm --prefix tests/sql ci
npm --prefix tests/sql run test:managed-interclub
npm --prefix apps/web run test:component
npm --prefix apps/web run build
npm --prefix apps/web run test:managed-interclub-browser
```

The browser journey runs real Next and FastAPI against the actual migration and RPC in isolated PGlite. Auth is synthetic only in the test fixture; all external browser requests are blocked. It checks organizer setup, anonymous host access, mobile layout, PDF printing, autosave, reload persistence, concurrent scorer conflicts, submission, approval, public standings and revocation. CI uses the built Next app and uploads screenshots, the paper packet and a result record.

Staging migration: `20261114223140_interclub_organizer_managed.sql`. Follow the repository's exact-SHA staging handoff requirements before reporting deployment readiness.

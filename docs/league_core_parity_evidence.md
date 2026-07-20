# League core parity evidence

This slice makes the `league_manager`, `league_printout`, and `top_players_printable`
surfaces automated-ready. It does not change their parity-matrix status; manual staging
acceptance remains consolidated in the final evidence PR.

## Automated contract

- League detail returns Python-derived lifecycle actions, settings mode, roster lock,
  validation errors, and non-blocking operational warnings.
- Create, duplicate, lifecycle, settings, and roster mutations require an authenticated
  club role with `manage_matches`, an exact confirmation phrase, and the FastAPI-only
  `SUPABASE_SERVICE_ROLE_KEY`.
- Roster writes are club/league/player scoped, closed-league safe, history preserving,
  stale-state aware, audit attributed, and compensated when staging requires an audit
  row but the audit write fails.
- Settings, roster, and lifecycle writes all reject inconsistent `status`/`is_active`
  state before mutation.
- League print data is an authenticated, read-only API projection. Weekly rating leaders
  use stored start/end snapshots when present and the canonical Python rating function
  when a legacy row lacks complete snapshots. Win leaders and configured season Top
  Performers are separate, labeled lists.
- Top Active Players uses the previous UTC calendar month and requires 10 scored games.
  Inactive players and current-month matches do not leak into the result.
- Both Next routes include explicit print CSS, table-header repetition, row/page-break
  controls, and a browser `Print or save PDF` handoff.

## Automated checks

```bash
pytest -q \
  tests/test_admin_league_manager_service.py \
  tests/test_admin_league_manager_lifecycle.py \
  tests/test_api_contract_admin_league_manager_roster.py \
  tests/test_api_contract_admin_league_manager_settings.py \
  tests/test_admin_league_print_service.py \
  tests/test_league_core_static_contract.py

cd apps/web
npx tsc --noEmit
npm run build
npx playwright test e2e/league-core.staging.spec.ts
```

The authenticated Playwright read-only check is enabled by
`JUPR_STAGING_ADMIN_ACCESS_TOKEN`; it never submits a mutation. The final manual session
will print/save a representative league night and previous-month ranking, approve page
breaks, and exercise a disposable league lifecycle with its documented recovery path.

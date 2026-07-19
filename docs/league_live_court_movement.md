# League Live court movement

League Manager Live persists a resumable league night, but FastAPI/Python is the sole authority for roster seeding, bench selection, score aggregation, player ranking, court movement, and next-round state. The Next browser renders the returned plan and may request reviewed overrides; it does not calculate or submit movement JSON.

## Movement rule

For every scored round, Python:

1. verifies that every scored match contains four distinct active players from one current court;
2. ranks each court by round wins, point differential, points scored, original slot, and stable player ID;
3. moves the top player on every court except Court 1 up one court;
4. moves the bottom player on every court except the lowest court down one court; and
5. keeps everyone else on the same court.

The stable tie breakers make repeated plans deterministic. `POST /admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan` returns the Python plan and its SHA-256 `operation_key`. The all-match submit endpoint recomputes the plan before publishing and accepts it only when the operation key still matches. Publish-only metadata such as deterministic context IDs does not alter movement identity.

## Roster and bench contract

- `POST /admin/clubs/{club_id}/league-manager/live/roster-suggestion` verifies each player against the club-scoped `players` table, creates only four- or five-player courts, and returns explicit active and bench rows with stable player IDs.
- An operator may select a different bench only when the requested count still fits the court capacity. A non-default bench requires a reason of at least 10 characters.
- Between rounds, an operator may add a verified club player or substitute one verified club player for an active player. Python rebuilds and validates the next roster and courts.
- Manual movement targets are reviewed by Python. Every target must be an existing court, every resulting court must contain four or five players, and a changed target requires a reason of at least 10 characters.

## Concurrency, idempotency, and recovery

- Snapshot and round-plan requests carry the session's `expected_updated_at`. Stale browser state receives HTTP 409 and must be reloaded.
- A complete-round publish carries `expected_operation_key`, `expected_match_count`, and a durable idempotency key. FastAPI refuses missing scores, stale versions, and changed plans before audit or official-match mutation.
- Each match slot receives a deterministic context. Before retrying a `publishing`/`retryable` operation, FastAPI reconciles all existing contexts, blocks partial context sets, and calls Match Uploader only when none exist.
- `league_live_rounds.operation_key` and `league_live_publish_operations` preserve snapshot and publish identity. Retrying a completed operation returns `idempotent_replay: true` without a second official match.
- If official scores exist but snapshot save or completion audit failed, reload the operation and use Reconcile. The error explicitly says the operation may have completed and forbids a blind retry.
- A partial publish remains `recovery_required`. After Match Log soft-exclusion and Replay History recovery, the compensation endpoint verifies zero active contexts and records the supplied recovery reference.

## Security and rollout

The League Live tables are private server data. The canonical Supabase migration enables RLS, revokes `public`, `anon`, and `authenticated`, and grants table access only to `service_role`. `SUPABASE_SERVICE_ROLE_KEY` stays on FastAPI and is never exposed to Next or a browser.

Planning fails closed unless `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER=1` and staging-only `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN=1` are enabled. Official submit additionally requires `JUPR_ENV=staging`, `JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT=1`, the service role, canonical match context/deletion columns, and both League Live migrations. Production cannot enable submit through the flag alone. Streamlit League Manager remains the operator fallback until the manual staging book is accepted; the parity matrix remains `Partial` until then.

# Player Editor activation candidate

The production Players screen is disabled by `JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR=0` in `fly.toml`.
This focused candidate enables the existing authenticated editor and improves its rating correction workflow.

- Admins can search the full club roster, add a player, and save name, current/starting JUPR, active status, and existing league-rating edits.
- The roster now pages through every player using stable name/ID ordering, including duplicate names.
- Profile saves send only changed fields. A current-rating correction preserves the exact historical starting rating, wins, losses, and match count.
- Rating inputs support three decimal places, including 4.408. An absent starting rating does not block editing another field.
- Social identity loading failures no longer discard the successfully loaded player roster.
- New production merges stay disabled by the separate `JUPR_ENABLE_NEXT_ADMIN_PLAYER_MERGE=0` switch. Existing staging merge behavior and recovery endpoints remain available. The merge service enforces the switch; hiding the UI is not the security boundary.

No schema migration or player-data change is included. In particular, this candidate does not change Carrie Blair's record.

## Validation

- 45 focused Python service/API/migration-contract tests passed, including production-mode profile create/update, cross-club roster scope, 1,201-player pagination, rating-only preservation, missing starting ratings, and production merge denial.
- React form tests passed for saving 4.408, preserving unchanged fields, no-op saves, missing historical ratings, and roster loading during a social-service error.
- Existing player-search and guarded-operation recovery checks passed.
- TypeScript checking and the Next.js optimized build passed.
- Live production acceptance is pending deployment approval; no production data was used for mutation tests.

## Release boundary

Feature PR targets `staging` under AGENTS.md. Production publication still requires Joe's explicit approval.
After approval, apply only this patch onto the then-current `rollback-feb8` branch, preserving its direct production fixes; do not merge all of staging.
Use the repository's exact-candidate API deployment and web deployment sequence. Do not alter the release trigger before approval.

After deployment, verify `/admin/players` loads for the authorized Tres administrator and the API reports `enabled=true`, `merge_enabled=false`.
Keep the existing profile fingerprint, idempotency, role, club-scope, and audit protections. Revert the editor flag to `0` if activation fails; this does not undo any explicitly saved player edits.

# Automatic player update delivery

Published league rounds, approved generator sessions, direct results, challenge
ladder results and tournament publications queue updates for active verified
subscriptions. Each automatic update records the exact committed match IDs and
the relevant award jobs. Retrying publication reuses the same session identity.

The production API starts the email scheduler on startup. It checks pending
session updates every 60 seconds, selecting at most 25 per pass. The publisher
and scheduler share `session_player_update_operation_key`; manual previews and
legacy rows without saved games are excluded. A rotating scan means a session
waiting for awards cannot indefinitely block later sessions.

The shared sender rechecks the subscription and claims each selected row with
its expected row version before SMTP. It waits for the session's award jobs;
the automatic worker can advance one pending badge job in the background per
pass. No badge processing is added to page loads. Failed and uncertain SMTP
attempts are not automatically repeated. An uncertain delivery remains
`sending` for reconciliation because the provider may already have accepted it.

On controlled shutdown, the API stops claiming further messages and allows its
current email and run log to finish, with a 240-second application drain and a
300-second Fly shutdown allowance. Abrupt host failures can still leave a
`sending` row; reconcile its `delivery_attempt_id` with the provider's Message-ID
before retrying. Never infer non-delivery from a missing `sent_at` alone.

## Verify operation

- `/health` exposes `write_prerequisites.player_update_worker_running` and an
  address-free `player_update_delivery` heartbeat: last start/completion, last
  successful send, last polling exception type and the latest batch counts.
- `worker_run_log` records every delivery batch and its sent/skipped/error/
  uncertain/award-wait totals. Batches with delivery or badge failures are
  marked failed instead of reporting success.
- `player_profile_update_outbox.sent_at`, `delivery_mode=live` and
  `provider_message_id` record successful handoff to SMTP. These are not proof
  of inbox placement; provider delivery/bounce records are separate.

Production requires the existing app/project identity, write policy, auto-send
and live-email flags, SMTP configuration and required worker logging. Staging
keeps its existing dry-run posture and never starts the production scheduler.
The API contract workflow runs the queue, session content, delivery, staging
email safety and publication-caller regressions on subsequent changes.

## Recover a backlog

Before enabling a repaired sender, inspect pending rows against their saved
match IDs, active subscriptions and exact award jobs. Reconcile excluded games
or overlapping sessions rather than resetting all statuses. Valid pending
automatic rows need no rewrite: the scheduler processes them after deployment.
Never reset sent rows or uncertain sends just to make the queue look empty.

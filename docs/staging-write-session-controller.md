# Protected staging write-session controller

This controller lets the connected GitHub agent open, advance, and close a
short staging write-test window without asking Joe to run a GitHub Actions
form. It never targets production and never enables more than one reviewed
write wave.

## Control issue

The only control plane is locked owner-authored issue `#1062`,
`Protected staging write session control`. The agent updates the fenced YAML
body and either reopens the issue for `open` or edits the already-open issue for
`advance` or `close`.

Joe does not edit this issue. He asks for the next test window in ChatGPT; the
agent resolves the current staging candidate, writes the exact command, and
reports the controller's status comment.

An `open` body is:

```yaml
command: open
candidate_sha: 0000000000000000000000000000000000000000
expected_write_wave: none
write_wave: league-manager
lease_started_at: 2026-07-26T18:00:00Z
lease_expires_at: 2026-07-26T18:30:00Z
session_nonce: 00000000-0000-4000-8000-000000000000
```

The agent replaces every example value. The candidate must be the current full
`staging` SHA. The timestamps use UTC seconds and define a lease of 5–60
minutes. The nonce is a new UUIDv4 and remains unchanged through every advance
and close in that session.

An `advance` body is:

```yaml
command: advance
candidate_sha: 0000000000000000000000000000000000000000
expected_write_wave: league-manager
write_wave: league-awards
lease_started_at: 2026-07-26T18:30:00Z
lease_expires_at: 2026-07-26T19:00:00Z
session_nonce: 00000000-0000-4000-8000-000000000000
```

Advance always deploys and verifies `none` before it deploys the next wave.
There is no active-wave-to-active-wave path. The issue edit must continue the
same candidate and nonce, and its prior body must name the new command's
`expected_write_wave`.

A `close` body is:

```yaml
command: close
candidate_sha: 0000000000000000000000000000000000000000
expected_write_wave: league-awards
write_wave: none
session_nonce: 00000000-0000-4000-8000-000000000000
```

One owner body edit is sufficient to close an open session. It must preserve
the active session's candidate and nonce and name the prior active wave as
`expected_write_wave`. The controller
deploys and verifies `none`, comments the safe result, and closes the issue.

## Allowlisted waves

The controller accepts the existing isolated waves from
`scripts/staging_write_waves.py`:

- `public-intake-auth`
- `communications`
- `match-player`
- `match-exclusion-recovery`
- `support-requests`
- `league-manager`
- `league-awards`
- `league-live-domain`
- `league-live-submit`
- `badge-diagnostics`
- `admin-tools`
- `challenge-ladder`
- `moneyball`
- `jupr-live`
- `public-live`
- `tournament-mutations`
- `tournament-setup`
- `tournament-registration`
- `tournament-operations`
- `tournament-official-publish`
- `tournament-email-handoff`
- `tournament-live`
- `tournament-live-official-publish`

`none` is a close target, not an open wave. There is no `all` wave. Dormant
direct-singles and tournament-import flags remain off.

## Safety behavior

Before opening or advancing, the controller requires:

- exact owner, repository, locked issue number/title, and event action;
- a fresh canonical body with no unknown fields;
- the current 40-character staging SHA;
- the expected current Fly wave and its exact controlled-flag fingerprint;
- the exact staging Vercel, Fly, and Supabase identities;
- service-role, API-audit, and worker-log readiness;
- `email_mode=dry_run`, live player-update email disabled, and the public-live
  production override disabled.

The transition jobs share the staging evidence mutex. Fly deployments also use
the existing global Fly mutex. An advance is `current -> none -> next`.

Each open or advance run starts an expiry watcher without holding the staging
mutex. At expiry it reacquires the mutex, rechecks the exact issue body and
nonce, restores `none`, and closes only if that lease is still current. An older
watcher cannot close a newer lease.

The scheduled safety workflow runs at minute `7` and `37`. It skips restoration
only when issue `#1062` is open and locked, the self-contained lease is valid
and unexpired, its candidate is current staging, and Fly health exactly matches
one bounded state for that command: `none` or target during open; expected,
`none`, or target during advance. The full controlled-flag fingerprint and
dry-run-email contract must match whichever state is current. Otherwise it restores `none` and
closes the issue. A failed or cancelled controller run always requests
restoration; the Fly mutex puts that recovery after any surviving deployment.

## Status comments

The workflow comments these milestones on issue `#1062`:

- command accepted;
- write session ready, with wave, candidate, expiry, and deployment run;
- close or expiry restored to `none`;
- failed transition restored safely, or a retry is required.

No token, password, service-role value, user email, or fixture payload is
included.

## Fixture boundary

This controller manages runtime authorization and its safe restoration. It does
not invent or broadly delete business fixtures. Each write-test packet must
still name its disposable records, versions, idempotency keys, authoritative
readback, truthful inverse or retained terminal evidence, and exact cleanup
owner. Existing Tournament Live and Match Log exclusion fixture automation can
run inside their matching waves; the remaining fixture packages are separate
work.

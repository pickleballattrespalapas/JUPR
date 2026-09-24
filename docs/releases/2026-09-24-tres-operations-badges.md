# Tres Palapas operations and badges release

Approved scope: badges, admin home and personal notifications, recap management,
generator result submission and approval, season leaderboards, and the existing
club's workspace foundation. Interclub leagues, new-club setup, commercial
onboarding, invitations for new clubs, and the website builder remain deferred.

## Baselines and preservation

- Production parent: `faf65d9efe8a9a346dab38aefcb4a64f501645bc` (`rollback-feb8`).
- Selected staging source: `92a288f7f1ab2fa7ba753cf17d913a5c33e51ef2`.
- Built on production with a selective squash; deferred staging history is not
  marked merged.
- Retains production registration editing/cancellation, gender eligibility
  exceptions, partner search and invitations, rating precision and locked
  profiles, previews, mobile emails, and tournament/league operations.
- Preserves every existing production feature gate and migration. Adds only the
  shell, weekly recap, badge diagnostics and JUPR Live feature flags.
- Badge and generator writes additionally require the exact production app,
  project, write posture, and `tres_palapas` club.
- Leaderboard publishing has its own service-only, audited settings transaction;
  it does not depend on club onboarding or website publication.

## Database changes

Apply these semantic migration names in the production contract order:

1. `club_staff_scopes`
2. `admin_personal_notifications`
3. `badge_reactivation_and_admin_seasons`
4. `program_badge_expansion`
5. `program_badge_reconciliation_actor`
6. `program_badge_round_robin_state`
7. `club_leaderboard_settings`

These changes are additive to existing production data. Do not use a timestamp
only database push: production's historical ledger has intentionally different
timestamps. Use the reviewed semantic-name migration contract.

## Badge activation

Read-only production preview on September 24 found 7,168 existing awards and:

| Group | New awards | Players |
| --- | ---: | ---: |
| Restored automatic badges | 671 | 138 |
| Program badges | 55 | 29 |

Player groups may overlap. There are 14 historical program review items whose
completion/result evidence is insufficient; the engine skips inferred trophies.
No seasons are configured yet. Community recognition remains an explicit admin
action. Existing manual revocations and legacy awards remain authoritative.

The deployment runs `python -m scripts.activate_operations_badges --apply` only
after production attestation succeeds. It recomputes from current data, records
an activation audit, inserts missing semantic identities, checks that existing
awards are unchanged, and safely recognizes a previously completed activation.
Final counts may change if real results arrive between preview and deployment.

## Verification

- Next production build and TypeScript checks passed.
- Full web component suite passed, including existing registration, partner,
  search and email coverage and the selected admin changes.
- Required API CI groups and selected release suites were exercised locally;
  stale scope assertions were corrected and their focused rerun passed.
- Staging rollback-only SQL checks passed for badge administration, program
  reconciliation, settings save/publish/discard, stale revisions, authorization,
  retries, revocations, audit records and service-only privileges.
- Browser notification tests run in GitHub CI and preserve desktop/mobile
  evidence. Local Chromium download was unavailable in the execution workspace.
- The entire legacy repository suite cannot collect because the unchanged
  Streamlit tournament-registration module has an existing indentation error.
  The API deployment gates and relevant production regression suites are used.

## Deployment and recovery

Require successful API, web and browser checks on the release candidate. Apply
the seven reviewed migrations, merge the selected candidate into production,
then use the existing trigger-only production release mechanism. Its schema
window and runtime checks retain their existing guarded promotion/recovery flow.
Verify the production API and web SHAs, public tournament/registration/player
reads, existing feature gates, and the badge activation artifact after release.

Do not roll back additive migrations or delete badges as a recovery shortcut.
Restore the last verified application/flag profile using the production workflow
if activation or runtime attestation fails, and inspect its recorded evidence.

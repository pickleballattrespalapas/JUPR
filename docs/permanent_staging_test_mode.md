# Fail-closed staging test mode

Staging is read-only at rest. Automatic deployments must preserve that posture;
business writes require one explicitly approved, least-privilege wave.

## Normal staging posture

- `JUPR_STAGING_WRITE_WAVE=none`
- Every reviewed staging mutation gate is disabled.
- Read-only public and authenticated staging flows remain available.
- A manual staging deployment may select one named write wave only for an
  explicitly approved acceptance action, then must restore `none`.
- Admin pages continue to require admin authentication.
- API audit logs, idempotency, transaction guards, recovery locks, and confirmation dialogs remain in place.
- Email remains `dry_run`; live player-update email is disabled.
- Dedicated public-live staging secrets take precedence when configured. When both are absent, Fly derives a stable, domain-separated pair from the existing staging service-role secret without printing or committing either value.

## Hard boundaries retained

- Fly target remains `juprleagues-api-staging`.
- Supabase target remains `sijpxjxvdtrehmqvirfi`.
- Production Supabase `dnoockbwfenunhcibwfn` is never a target.
- Production deployment is never triggered by staging workflows.
- `JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION` remains disabled.

## Emergency stop

`Staging Emergency Write Disable` is a manual owner-only workflow that deploys
`write_wave=none`. Normal staging deployments also default to `none`; they never
reopen a business-write wave automatically.

Admin status fetches use no-store responses so pages cannot remain stuck on a previously deployed guarded state. This implementation converted 14 cached admin fetch call(s).

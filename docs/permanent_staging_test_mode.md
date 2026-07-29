# Permanent staging test mode

Staging is intentionally writable at all times for acceptance testing. The former issue-controlled, 60-minute, one-wave-at-a-time lease system is retired.

## Normal staging posture

- `JUPR_STAGING_WRITE_WAVE=open`
- Every reviewed staging mutation gate is enabled.
- Public and authenticated staging test flows remain available without reopening a window.
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

`Staging Emergency Write Disable` is a manual owner-only workflow that can deploy `write_wave=none`. The next normal staging deployment restores permanent `open` mode.

Admin status fetches use no-store responses so pages cannot remain stuck on a previously deployed guarded state. This implementation converted 14 cached admin fetch call(s).

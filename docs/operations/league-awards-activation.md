# League Awards availability

The Fall Ladder League 2026 screenshots show a draft league with no configured
award categories. The production branch has
`JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=0`, and the old Awards service rejected
every production mutation even if that flag was enabled. The review page also
described every unavailable state as a staging test restriction.

The fix makes the Awards service and League Manager status use one availability
check. It requires the League Manager and Awards flags, server credentials, an
explicit supported environment, and the production write policy when running in
production. Staging also requires its Awards write wave and League Manager write
gate. Existing route authentication, club permissions, database transactions,
version checks, audit logging, and badge verification remain in place.

The settings and review panels display the API's explanation when editing is
disabled. Settings also checks credential readiness and explains when setup is
locked by league progress or final award review.

## Production activation

The staging change leaves the checked-in production configuration disabled.
Production activation requires Joe's explicit approval under `AGENTS.md`.

After approval, release only this change from the current `rollback-feb8` branch,
preserving its production configuration. Set
`JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1` in the production Fly configuration
and include that flag in the production verifier's release feature profile.
Keep the rollback baseline profile unchanged. Deploy the API and web candidate,
then verify the exact deployed revision and Awards availability.

This change adds no database migration. Read-only inspection of staging confirmed
the award configuration save function, its service-role execute grant, the result
set table, and the configuration version column. Check the same prerequisites
in production as part of the approved release; do not change a real league or
publish its awards just to test deployment.

## Verification

API tests exercise setup save/reload with production runtime settings, selected
club isolation, permission denial, disabled gates, persisted review, override
reasons, verified badge publishing, retries, and archive recovery. Component tests
exercise category selection, places, minimum games, reload, disabled editing, and
locked setup. These tests use local fixtures rather than production data.

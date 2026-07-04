# Public Launch Decisions

This document records owner decisions for the first Pickleball Club Sandwich public website draft.

## Brand and domains

- Primary SaaS name: Pickleball Club Sandwich.
- Primary public domain: `pickleballclubsandwich.com`.
- Primary www alias: `www.pickleballclubsandwich.com`.
- Transitional alias: `juprleagues.com` and `www.juprleagues.com`.
- Continue using internal JUPR package/service names until a separate technical rename plan is reviewed.
- Public copy should move toward Pickleball Club Sandwich first, then internal and admin naming can follow in later slices.

## Legal pages

- `/privacy` and `/terms` should use formal first-party copy now instead of operational placeholders.
- The legal copy can still be reviewed and amended before broad launch.

## Support routing

- Public support email: `joe@juprleagues.com`.
- Support and contact routes should route general support, correction routing, and tournament help there unless a more specific workflow exists.

## Tournament board interest notifications

- When a token-verified public board interest action is submitted, notify both:
  - the player whose board entry was selected; and
  - the organizer inbox.
- Organizer inbox resolution should prefer `JUPR_TOURNAMENT_ORGANIZER_EMAIL`, then `JUPR_SUPPORT_EMAIL`, then `joe@juprleagues.com`.
- Use existing email safety modes: `dry_run`, `staging_redirect`, and `live`.
- Keep the board action pending-only. Do not auto-confirm a team from the public action.

## Staging smoke

- Use `--allow-live-unconfigured` for first staging smoke while live-session migrations/grants are still being verified.
- Remove `--allow-live-unconfigured` once live sessions are fully configured in staging.

## Custom-domain cutover

- Use `pickleballclubsandwich.com` as canonical when Vercel domain verification is complete.
- Keep `juprleagues.com` live as a transitional alias during the naming migration.
- Keep Streamlit available as fallback until both the primary domain and transitional alias pass public smoke.

## Immediate next work

- Continue public copy cleanup toward Pickleball Club Sandwich.
- Avoid risky internal renames until the public domain cutover is stable.
- Run full staging smoke when staging FastAPI and Vercel URLs are available.

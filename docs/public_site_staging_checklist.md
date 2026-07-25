# Public Site Staging Checklist

Use this checklist to review the Next/Vercel public website candidate before any
custom-domain cutover. The candidate must come from canonical branch `staging`;
`Test` is a legacy/deprecated staging branch and is not an acceptance source.

## Required environment

Backend staging:

```text
JUPR_ENV=staging
JUPR_API_BASE_URL=<staging FastAPI base URL>
SUPABASE_URL=<staging Supabase project URL>
SUPABASE_SERVICE_ROLE_KEY=<staging service role key>
SUPABASE_ANON_KEY=<staging anon key>
JUPR_EMAIL_MODE=staging_redirect or dry_run
JUPR_STAGING_EMAIL_REDIRECT_TO=<safe test inbox when staging_redirect is used>
JUPR_TOURNAMENT_ORGANIZER_EMAIL=joe@juprleagues.com
JUPR_SUPPORT_EMAIL=joe@juprleagues.com
JUPR_PUBLIC_SUPPORT_RATE_LIMIT_PER_HOUR=5
```

Frontend staging:

```text
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging FastAPI base URL>
NEXT_PUBLIC_JUPR_WEB_BASE_URL=<staging Vercel URL>
```

Production public web URL after domain verification:

```text
NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com
```

Do not put service-role credentials in Vercel frontend variables.

## Domain targets

Primary public SaaS domain:

```text
https://pickleballclubsandwich.com
https://www.pickleballclubsandwich.com
```

Transitional alias while public naming migration completes:

```text
https://juprleagues.com
https://www.juprleagues.com
```

FastAPI CORS should allow all four public origins during the transition.

Read-only snapshot on 2026-07-24:

- Both apex domains returned HTTP 200.
- Both `www` hosts returned HTTP 502 and were not assigned to the Vercel project.
- Production FastAPI CORS preflight returned the exact requesting origin for all
  four allowed origins.

This is inventory, not cutover approval. Joe or the domain owner must add/verify
the two `www` aliases and DNS later; no DNS or production-domain setting was
changed during staging preparation.

## CI gates

Every public web PR should pass:

- Migration Source Guard.
- API Contract Tests.
- Next Web Build when web files change.

## Canonical smoke and diagnostics

The evidence-grade gate is the manually dispatched GitHub Actions workflow
`Staging Smoke`. Run it from the final deployed `staging` SHA only after Fly has
been restored to `write_wave=none` and `/health` reports every controlled write
flag false. Vercel and Fly must attest the same SHA, deployment identity, staging
Supabase/Auth origin, and immutable Vercel candidate. The workflow rejects skips,
flakes, or count drift across its 56-test public-read manifest and its five-test
guided Tournament Setup browser contract.

This local command is useful for diagnosis after FastAPI and Vercel are deployed,
but it is not the canonical acceptance gate:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <staging FastAPI base URL> \
  --web-base-url <staging Vercel URL>
```

The legacy `Public Web Smoke` workflow is also noncanonical. A green diagnostic
does not replace the strict browser manifest and identity binding in `Staging Smoke`.

Run the same smoke against both public domains after production deploy:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://pickleballclubsandwich.com

python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://juprleagues.com
```

## Public route review

Review these pages on staging and the primary public domain:

| Area | Route | Review notes |
|---|---|---|
| Home | `/` | Product copy, primary links, graceful API fallback. |
| Route map | `/site-map` | Full click-through coverage. |
| Club | `/clubs/tres-palapas` | Club name, cards, public links. |
| Live | `/clubs/tres-palapas/live` | Empty state and session detail links. |
| Leaderboards | `/clubs/tres-palapas/leaderboards` | Public-safe player fields and league filters. |
| Match Explorer | `/clubs/tres-palapas/match-explorer` | Player selectors and preview errors. |
| League Results | `/clubs/tres-palapas/league-results` | Standings, summaries, empty states. |
| Badge Codex | `/clubs/tres-palapas/badge-codex` | Badge list, earners, load-more state. |
| Challenge Ladder | `/clubs/tres-palapas/challenge-ladder` | Ladder tiers and rules copy. |
| Weekly Recap | `/clubs/tres-palapas/weekly-recap` | Published recap, print view, PDF link. |
| Tournament Registration | `/clubs/tres-palapas/tournament-registration` | Intake form, edit-link request, closed/open states. |
| Registration Confirmation | `/clubs/tres-palapas/tournament-registration/confirmation` | Confirmation copy and missing-id state. |
| Registration Edit | `/clubs/tres-palapas/tournament-registration/edit` | Invalid token state and valid-token edit flow. |
| Tournament Roster | `/clubs/tres-palapas/tournament-roster` | Public-safe roster fields only. |
| Tournament Board | `/clubs/tres-palapas/tournament-partner-board` | Public-safe board entries and token-gated action state. |
| Players | `/clubs/tres-palapas/players` | Active default, visible search, status/sort/paging, empty/error states, stable profile/row links. |
| Player Profile | `/clubs/tres-palapas/players/<id>` | Rating trend and Singles/Doubles breakdown, full/recent history, badges/trophies, partner/rival facts, Club Social aggregates, verified-update/privacy entry, public-field denylist. |
| Player Matches | `/clubs/tres-palapas/players/<id>/matches` | Match list and empty state. |
| Match List | `/clubs/tres-palapas/matches` | Filters and match detail links. |
| Match Detail | `/clubs/tres-palapas/matches/<id>` | Public-safe match fields. |
| Ratings Explainer | `/how-ratings-work` | Product copy accuracy. |
| FAQ | `/faq` | Public support accuracy. |
| Support | `/support` and `/contact` | Email link routes to `joe@juprleagues.com`; blank form renders without creating a request. |
| Data Corrections | `/data-corrections` | Instructions and blank staff-review intake render; no ratings, match, player, or tournament mutation occurs. |
| Profile Privacy | `/profile-privacy` | Instructions, request-type selector, and blank staff-review form render; no public profile/history change occurs. |
| Email Preferences | `/email-preferences` without a token | Expected safe state is `Preference link not found`; no preference/subscription mutation occurs. |
| Privacy | `/privacy` | Formal first-party policy copy present; review final language. |
| Terms | `/terms` | Formal first-party terms copy present; review final language. |
| Fallback | unknown route | Public fallback page links back to safe routes. |

## Manual write-flow review

Use staging data only.

At rest, Fly must report `write_wave=none`. A write review requires a separately
approved fixture/recovery packet and a manually dispatched Fly release for exactly
one least-privilege wave. Record that release, perform only the packet's named
action, verify authoritative readback/audit evidence, then deploy and record a new
`none` release. Do not enable all write flags together and do not run canonical
`Staging Smoke` while any write wave is active.

Do not infer write approval from this checklist. When a later packet explicitly
authorizes the corresponding route, the possible checks are:

- submit one disposable intake and an exact retry to prove deduplication;
- confirm registration pages render without private staff fields before any registration write;
- request an edit link and verify the response does not reveal whether the email matched;
- open a valid edit link, verify email is locked, then update only the disposable registration;
- verify roster/board projections never expose phone or email;
- exercise partner-board notification only with approved staging recipients and the configured non-delivery email mode.

## Decisions now encoded

- Privacy and Terms use formal first-party copy instead of placeholders.
- Public support routes to `joe@juprleagues.com`.
- Public tournament board interest actions notify both the affected player and organizer.
- Live-session routes and migrations are part of the required candidate; public
  diagnostics and canonical Staging Smoke fail closed when they are unavailable.
- `pickleballclubsandwich.com` is the primary public SaaS domain; `juprleagues.com` remains a transitional alias.

## Remaining deployment tasks

- Merge and deploy the repair candidate, apply/verify the reviewed 38-name
  staging migration inventory including `singles_replay_recovery`, and rerun the
  exact-SHA public/admin read evidence. The superseded `7cedb81` artifacts are
  diagnostic only.
- Add and verify both `www` aliases in Vercel/DNS; the two apex domains are
  currently served, while the `www` hosts returned HTTP 502 on 2026-07-24.
- Reconfirm FastAPI CORS at cutover. Read-only preflight checks passed for the
  primary and transitional domains plus both `www` aliases on 2026-07-24.
- Set `NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com` for production.
- Run canonical `Staging Smoke` against the final same-SHA staging candidate; use diagnostic smoke against public domains only after a separately approved production cutover.
- Obtain legal approval for Privacy and Terms, prove password recovery and
  unsubscribe/preferences through an approved test inbox, decide on Supabase
  leaked-password protection, and establish minimum alerts.
- Keep Streamlit available as fallback until the custom-domain smoke passes.

The full 45-row administrative parity book governs later Streamlit retirement;
it is not a reason to make billing, self-serve onboarding, a second-club pilot,
or still-guarded admin replacements block the initial public Tres Palapas
launch. Legal/domain/monitoring/approved-inbox prerequisites still apply.

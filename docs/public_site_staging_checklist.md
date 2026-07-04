# Public Site Staging Checklist

Use this checklist to review the first full Next/Vercel public website draft before or immediately after custom-domain cutover.

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

## CI gates

Every public web PR should pass:

- Migration Source Guard.
- API Contract Tests.
- Next Web Build when web files change.

## Smoke command

Run this against staging after FastAPI and Vercel are deployed:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <staging FastAPI base URL> \
  --web-base-url <staging Vercel URL> \
  --allow-live-unconfigured
```

`--allow-live-unconfigured` is allowed for the first public draft while live-session staging migrations/grants are still being verified. Remove it once live sessions are fully configured.

Run the same smoke against both public domains after production deploy:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://pickleballclubsandwich.com \
  --allow-live-unconfigured

python scripts/smoke_public_web.py \
  --api-base-url <public FastAPI base URL> \
  --web-base-url https://juprleagues.com \
  --allow-live-unconfigured
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
| Players | `/clubs/tres-palapas/players` | Search, empty state, profile links. |
| Player Profile | `/clubs/tres-palapas/players/<id>` | Ratings, records, match links. |
| Player Matches | `/clubs/tres-palapas/players/<id>/matches` | Match list and empty state. |
| Match List | `/clubs/tres-palapas/matches` | Filters and match detail links. |
| Match Detail | `/clubs/tres-palapas/matches/<id>` | Public-safe match fields. |
| Ratings Explainer | `/how-ratings-work` | Product copy accuracy. |
| FAQ | `/faq` | Public support accuracy. |
| Support | `/support` and `/contact` | Email link routes to `joe@juprleagues.com`. |
| Data Corrections | `/data-corrections` | Checklist language and support handoff. |
| Privacy | `/privacy` | Formal first-party policy copy present; review final language. |
| Terms | `/terms` | Formal first-party terms copy present; review final language. |
| Fallback | unknown route | Public fallback page links back to safe routes. |

## Manual write-flow review

Use staging data only.

- Submit a tournament registration with one event.
- Confirm the registration page and confirmation page render without private staff fields.
- Request an edit link and confirm the response does not reveal whether the email matched.
- Open a valid edit link and confirm email is locked.
- Edit selections and confirm the registration updates the existing record.
- Open roster and board pages and confirm public entries do not expose phone or email.
- Submit a public tournament board interest action and confirm player plus organizer email handling uses the configured staging email mode.

## Decisions now encoded

- Privacy and Terms use formal first-party copy instead of placeholders.
- Public support routes to `joe@juprleagues.com`.
- Public tournament board interest actions notify both the affected player and organizer.
- First staging smoke may use `--allow-live-unconfigured`; remove it after live-session setup is complete.
- `pickleballclubsandwich.com` is the primary public SaaS domain; `juprleagues.com` remains a transitional alias.

## Remaining deployment tasks

- Confirm Vercel serves both primary and transitional domains.
- Confirm FastAPI CORS allows the primary and transitional domains plus their `www` aliases.
- Set `NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com` for production.
- Run smoke against staging, `pickleballclubsandwich.com`, and `juprleagues.com`.
- Keep Streamlit available as fallback until the custom-domain smoke passes.

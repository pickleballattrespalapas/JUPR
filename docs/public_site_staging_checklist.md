# Public Site Staging Checklist

Use this checklist to review the first full Next/Vercel public website draft before any custom-domain cutover.

## Required environment

Backend staging:

```text
JUPR_ENV=staging
JUPR_API_BASE_URL=<staging FastAPI base URL>
SUPABASE_URL=<staging Supabase project URL>
SUPABASE_SERVICE_ROLE_KEY=<staging service role key>
SUPABASE_ANON_KEY=<staging anon key>
JUPR_EMAIL_MODE=staging_redirect or dry_run
```

Frontend staging:

```text
NEXT_PUBLIC_JUPR_ENV=staging
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging FastAPI base URL>
NEXT_PUBLIC_JUPR_WEB_BASE_URL=<staging Vercel URL>
```

Do not put service-role credentials in Vercel frontend variables.

## CI gates

Every public web PR should pass:

- Migration Source Guard.
- API Contract Tests.
- Next Web Build.

## Smoke command

Run this against staging after FastAPI and Vercel are deployed:

```bash
python scripts/smoke_public_web.py \
  --api-base-url <staging FastAPI base URL> \
  --web-base-url <staging Vercel URL> \
  --allow-live-unconfigured
```

Remove `--allow-live-unconfigured` once live-session migrations and grants are ready in staging.

## Public route review

Review these pages on staging:

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
| Support | `/support` and `/contact` | Email link and correction routing. |
| Data Corrections | `/data-corrections` | Checklist language and support handoff. |
| Privacy | `/privacy` | Placeholder requires final legal approval. |
| Terms | `/terms` | Placeholder requires final legal approval. |
| Fallback | unknown route | Public fallback page links back to safe routes. |

## Manual write-flow review

Use staging data only.

- Submit a tournament registration with one event.
- Confirm the registration page and confirmation page render without private staff fields.
- Request an edit link and confirm the response does not reveal whether the email matched.
- Open a valid edit link and confirm email is locked.
- Edit selections and confirm the registration updates the existing record.
- Open roster and board pages and confirm public entries do not expose phone or email.

## Launch blockers requiring owner input

- Final legal approval for Privacy and Terms.
- Final support email and contact routing.
- Decision on whether public board interest actions should send notifications immediately or remain pending-only.
- Staging Vercel URL and staging FastAPI URL for final smoke.
- Custom-domain rollback instructions and DNS cutover timing.

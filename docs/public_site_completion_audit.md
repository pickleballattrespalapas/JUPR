# Public Site Completion Audit

Current working assessment for the first Pickleball Club Sandwich public website draft.

## Overall status

- First working public website draft: approximately 90% complete.
- Custom-domain launch readiness from the repository side: approximately 80% complete.
- Full Streamlit replacement: approximately 80% complete. Feature implementation is further ahead than operational readiness; guarded admin workflows still require isolated staging pilots before cutover.

## Public website coverage

| Area | Status | Notes |
|---|---|---|
| Home | Complete draft | Public copy moved toward Pickleball Club Sandwich. |
| Site map | Complete draft | Click-through public route map exists. |
| Club home | Complete draft | Public club cards and navigation exist. |
| Live sessions | Complete draft | Public live list/detail and edit-token scoring exist; staging may allow live setup 503 during smoke. |
| Leaderboards | Complete draft | Public ranking table exists. |
| Match Explorer | Complete draft | Public preview flow exists; rating math remains in Python/FastAPI. |
| League Results | Complete draft | Public standings/results summaries exist. |
| Badge Codex | Complete draft | Public badge metadata and earners exist. |
| Challenge Ladder | Complete draft | Public ladder/active challenge read side exists. |
| Weekly Recap | Complete draft | Published recap, print view, and PDF link exist. |
| Tournament Registration | Strong draft | Public intake and validation path exists. |
| Registration confirmation | Draft | Confirmation route exists. |
| Edit-link request | Strong draft | Non-enumerating request flow exists. |
| Tokenized registration edit | Strong draft | Server-verified edit route exists; email remains locked. |
| Tournament roster | Strong draft | Public-safe roster projection exists. |
| Tournament board | Draft | Public-safe board plus token-gated interest flow exists. |
| Player directory/profile/history | Complete draft | Public player and match surfaces exist. |
| Support/contact | Complete draft | Routes to `joe@juprleagues.com`. |
| Privacy/Terms | Partial | First-party privacy copy exists; Terms remains a placeholder pending owner/legal approval. |
| Data corrections | Complete draft | Staff-reviewed intake instructions exist; no public mutation. |

## Admin migration coverage

| Workflow | Status | Notes |
|---|---|---|
| Admin cockpit | Complete draft | Status and migration handoff route exists. |
| Match Log | Partial | Read, duplicate scan, guarded apply/cleanup exist. |
| Replay History | Partial | Guarded replay foundation exists. |
| Match Uploader | Partial/strong pilot | Manual batch, round-robin preview, new-player continue flow exist. |
| Player Editor | Guarded staging pilot | Roster/detail/create/basic update, merge preview/execute, social identity, and league-rating edits exist behind authenticated FastAPI boundaries. |
| League Manager | Guarded staging pilot | Draft creation/duplication, structured settings, lifecycle, roster, live rounds, print, and awards flows use authenticated FastAPI boundaries; Streamlit remains the production fallback. |
| Tournament admin/ops | Guarded staging pilot | Setup, registration management, draws, scoring, podiums, awards, live operation, and official publishing exist; full staging E2E remains required. |
| Challenge Ladder admin | Guarded staging pilot | Creation, clock/acceptance, read-only result preview, reviewed publish, official matches, rank movement, cancellation, forfeits, monthly passes, roster append/reactivation, audited tier moves, vacation/reinstate overrides, and computed tier-movement review exist. Whole-tier replacement remains Streamlit-only; centralized activity review is available in Admin Tools. |
| Weekly Recap admin | Guarded staging pilot | Draft generation, edits, preview, publish, and unpublish exist behind authenticated FastAPI boundaries. |

## Decisions already encoded

- Primary SaaS/domain name: Pickleball Club Sandwich.
- Primary domain: `pickleballclubsandwich.com`.
- Transitional alias: `juprleagues.com`.
- Public support email: `joe@juprleagues.com`.
- Public board interest notifies both selected player and organizer but remains pending-only.
- First smoke pass may use `--allow-live-unconfigured` until live-session setup is fully verified.

## Remaining repository-side work

- Continue copy cleanup away from legacy public naming where it appears in user-facing pages.
- Add deeper browser/E2E tests after staging URLs are stable.
- Validate organizer registration management and Partner Board review/accept/cancel behavior end to end in staging.
- Continue replacing public-facing naming while keeping internal package/env/table names stable until a separate technical rename plan exists.

## Remaining deployment validation

- Confirm Vercel production environment uses `NEXT_PUBLIC_JUPR_WEB_BASE_URL=https://pickleballclubsandwich.com`.
- Confirm FastAPI CORS allows `pickleballclubsandwich.com`, `www.pickleballclubsandwich.com`, `juprleagues.com`, and `www.juprleagues.com`.
- Run `scripts/smoke_public_web.py` against staging, then both public domains.
- Keep Streamlit available as the operational/admin fallback until domain smoke passes.

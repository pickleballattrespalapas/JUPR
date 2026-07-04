# Public Website First Draft

This is the current public-facing Next/Vercel route map for the first full Pickleball Club Sandwich website draft.

## Primary public routes

| Route | Purpose | Current status |
|---|---|---|
| `/` | Product home and public website entry point. | Draft |
| `/site-map` | Click-through route map for staging review. | Draft |
| `/clubs/tres-palapas` | Club home for Tres Palapas. | Draft |
| `/clubs/tres-palapas/live` | Public live-session list. | Draft |
| `/clubs/tres-palapas/live/[sessionKey]` | Public live-session detail and edit-token score entry. | Draft |
| `/clubs/tres-palapas/leaderboards` | Public club leaderboards. | Draft |
| `/clubs/tres-palapas/match-explorer` | Public match impact explorer. | Draft |
| `/clubs/tres-palapas/league-results` | Public league results. | Draft |
| `/clubs/tres-palapas/badge-codex` | Public badge catalog and earners. | Draft |
| `/clubs/tres-palapas/challenge-ladder` | Public challenge ladder. | Draft |
| `/clubs/tres-palapas/weekly-recap` | Public weekly recap. | Draft |
| `/clubs/tres-palapas/tournament-registration` | Public tournament intake and secure edit-link request. | Draft |
| `/clubs/tres-palapas/tournament-registration/confirmation` | Public registration confirmation. | Draft |
| `/clubs/tres-palapas/tournament-registration/edit` | Secure tokenized registration edit. | Draft |
| `/clubs/tres-palapas/tournament-roster` | Public-safe tournament roster. | Draft |
| `/clubs/tres-palapas/tournament-partner-board` | Public-safe tournament partner board with token-gated interest flow. | Draft |
| `/clubs/tres-palapas/players` | Public player directory. | Draft |
| `/clubs/tres-palapas/players/[playerId]` | Public player profile. | Draft |
| `/clubs/tres-palapas/players/[playerId]/matches` | Public player match history. | Draft |
| `/clubs/tres-palapas/matches` | Public match list. | Draft |
| `/clubs/tres-palapas/matches/[matchId]` | Public match detail. | Draft |
| `/how-ratings-work` | Public rating explainer. | Draft |
| `/faq` | Public FAQ. | Draft |
| `/privacy` | Formal first-party privacy policy copy. | Draft/legal review |
| `/terms` | Formal first-party terms copy. | Draft/legal review |
| `/support` | Support route using `joe@juprleagues.com`. | Draft |
| `/contact` | Alias to support route. | Draft |
| `/data-corrections` | Public data correction instructions using `joe@juprleagues.com`. | Draft |

## Public API routes used by the website

| Route | Purpose |
|---|---|
| `GET /clubs/{club_slug}` | Public club metadata. |
| `GET /clubs/{club_slug}/leaderboards` | Public leaderboards. |
| `GET /clubs/{club_slug}/league-results` | Public league results. |
| `GET /clubs/{club_slug}/badges` | Public badge catalog. |
| `GET /clubs/{club_slug}/challenge-ladder` | Public ladder data. |
| `GET /clubs/{club_slug}/weekly-recaps` | Public weekly recap list/detail. |
| `GET /clubs/{club_slug}/weekly-recaps/{week_start}/pdf` | Weekly recap PDF export. |
| `GET /clubs/{club_slug}/tournament-registration` | Tournament registration page model. |
| `POST /clubs/{club_slug}/tournament-registration` | Tournament registration submit. |
| `POST /clubs/{club_slug}/tournament-registration/edit-link/request` | Non-enumerating secure edit-link request. |
| `GET /clubs/{club_slug}/tournament-registration/edit` | Tokenized registration edit page model. |
| `POST /clubs/{club_slug}/tournament-registration/edit` | Tokenized registration edit submit. |
| `GET /clubs/{club_slug}/tournament-roster` | Public-safe tournament roster and board projection. |
| `POST /clubs/{club_slug}/tournament-registration/pairing-interest` | Token-gated tournament pairing-interest submit. |
| `GET /clubs/{club_slug}/players` | Public player directory. |
| `GET /clubs/{club_slug}/players/{player_id}` | Public player profile. |
| `GET /clubs/{club_slug}/matches` | Public match list. |
| `GET /clubs/{club_slug}/matches/{match_id}` | Public match detail. |
| `GET /clubs/{club_slug}/match-explorer` | Public Match Explorer context. |
| `GET /clubs/{club_slug}/match-explorer/preview` | Public Match Explorer preview. |
| `GET /clubs/{club_slug}/live-sessions` | Public live-session list. |
| `GET /clubs/{club_slug}/live-sessions/{session_key}` | Public live-session detail. |

## Draft guardrails

- Browser code does not read Supabase directly.
- Browser code does not receive service-role credentials.
- Rating, match-processing, registration persistence, roster projection, replay, and pairing-interest writes stay in Python/FastAPI.
- Public tournament roster and board views omit private contact fields.
- Public edit links are tokenized and verified server-side.
- Public board interest creates a pending record and notifies the selected player plus organizer without auto-confirming a team.
- Admin write surfaces remain guarded behind FastAPI feature flags and role checks.

## Review notes

The first draft is broad enough for route-by-route staging review. Remaining public-launch tasks are staging smoke on the configured domains, final review of legal/support copy, and continued public naming cleanup toward Pickleball Club Sandwich while preserving internal technical names until a separate rename plan is reviewed.

# Public multi-club websites — staging

This release implements the public design agreed in the questionnaire. It targets staging only.

## Visitor experience

- First visit: PCS introduction with **Find my club**, **Create a club**, and staff sign-in.
- Find my club: club-name search and an alphabetical, paginated directory of published, listed clubs.
- Return visit to `/`: open the last successfully visited club on this device. **Change club** opens the directory; **Powered by PCS** opens `/?welcome=1` without redirecting.
- Club pages lead with club name/logo, introduction, location and visitor information. Anyone may browse; there are no player accounts.
- Unlisted websites are available by direct link, excluded from PCS discovery and sitemap, and marked noindex. Unlisted means discoverability control, not authentication or a secret access token.
- Unpublished websites return not found, including club-scoped public APIs. Publication lookup outages fail closed rather than inventing a Tres identity.

## Administrator workflow

**Club website** is in the admin sidebar. The editor has introduction, pages/layout, stats/information, and preview tabs.

1. Edit the public identity, visitor information, logo and accent color.
2. Add custom pages, choose their addresses/navigation order, and add/reorder text, image, button, divider or club-link blocks. Choose widths (quarter to full row), alignment, background and spacing; blocks stack on phones.
3. Choose player-table statistics, profile sections, leaderboard columns, and registration/results information. These are presentation settings, not new data-access permissions; required registration fields remain present.
4. Save a draft. Preview desktop/mobile content and example display settings.
5. Publish when ready. Text, layout, logo and visibility changes stay separate from the live snapshot until publication. Unpublish retains the draft. Restore published version discards draft changes explicitly.

PNG/JPEG/WebP images up to 220 KB may be uploaded directly; larger images use HTTPS URLs. No arbitrary HTML/scripts are executed. Limits: 20 pages, 40 blocks per page, 2 MB website document. Club operational settings remain separate from website drafts.

**Create a club** accepts an existing verified administrator login or a new administrator account with email verification. Creation atomically binds the authenticated user to the new club as administrator. Existing slugs cannot be claimed. The website starts unpublished. Staging keeps email disabled: use an existing verified test login; no SMTP or verification bypass was added.

## Interclub league websites

From **Interclub seasons**, the organizer opens **Public schedule, results & standings**. Save/preview/publish a league schedule with accepted clubs, then add three-game encounter results per meet, division and club pairing. Scores follow games to 11, win by two, no cap. Published results determine encounter wins, game difference and point difference; equal totals remain tied.

Participating club pages link to the same whole-league website. No global interclub directory is added. Club slugs, contacts, invitations and internal rosters are not included in the league snapshot. Public league pages are noindex so participation does not index unlisted club names. Rosters remain per meet. Public result entry does not apply rating updates.

## Staging test data

| Club | Visibility | Distinguishing fixture |
| --- | --- | --- |
| La Ribera Pickelball Club | Listed | Blue accent and custom Visiting page |
| Cabo Test Club | Listed | Teal accent; singles and badges hidden |
| La Paz Test Club | Unlisted | Purple accent and direct-link access |

Existing players, matches and rosters are preserved. Each club has its own public snapshot and custom page. The **Three Club Isolation Test** league has a published schedule and empty results ready for score-entry testing. Fixture scripts are guarded and idempotent; they never reset later edits.

## Verification and operations

- Migrations: `20260916052824_public_club_sites.sql` and `20260916060108_public_club_directory_sort.sql`; applied only to staging Supabase `sijpxjxvdtrehmqvirfi`.
- New site/publication tables and all mutation functions are service-role only, with RLS and explicit revocation of anonymous/authenticated SQL access. Mutation RPCs recheck actor assignments under the existing staff lock; optimistic revisions reject stale saves/publications.
- Public routes expose only published snapshots. No-store responses prevent stale publication/visibility on the new surfaces.
- API contract coverage includes unsafe content rejection, unpublished guards, identity/club isolation, role revocation, conflict responses, signup and interclub standings validation.
- Component coverage exercises draft/preview/publish, duplicate actions, stale responses after club/account changes, token refresh, remembered-club fallback and existing-account club creation.
- `scripts/fixtures/verify_public_club_sites.sql` performs real database mutation checks inside a rolled-back transaction.
- Existing Supabase advisor warnings are unchanged. New tables deliberately have no client RLS policies because the verified API is the only entry point; direct grants are revoked. See the [RLS advisor explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).

Manual browser acceptance still needs the signed-in staging session. The browser available to the agent has Vercel login protection and cannot open local previews; automated component/API/build checks do not substitute for that acceptance.

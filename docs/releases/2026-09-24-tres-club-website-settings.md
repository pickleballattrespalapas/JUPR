# Tres Palapas club website settings

Joe requested the existing club website editing options in production and a clear
way to save leaderboard settings. This release extends the approved operations
release from production commit `c09837b0dc78facbfbc3b1af84396bb0d1c82f22`.

## Included

- **Club website**: introduction, visitor information, logo and accent color,
  custom pages and layouts, page visibility, displayed statistics, and preview.
- A saved draft stays separate from the published website. Restore, unpublish,
  and republish retain the existing revision and administrator checks.
- **Leaderboard settings** has its own sidebar entry and route at
  `/admin/leaderboard-settings`. A visible action bar remains on screen while
  editing. Save draft, then Publish settings applies the choices to the live
  leaderboard independently of the website draft.
- Public pages, navigation, metadata, and sitemaps follow the published website
  choices. Link-only sections remain accessible using their direct links.
- The existing tournament registration, cancellation, partner, gender review,
  rating, email, and rich-preview behavior is retained.

Club creation, club switching, commercial onboarding, interclub administration,
and interclub public pages remain excluded. Existing active clubs retain their
public identity, public sections, and search indexing during initialization.
Existing website documents and leaderboard settings are not overwritten.

## Database and deployment

`20260924181653_club_website_settings.sql` adds the website snapshot table,
service-only mutation RPC, and audit. It rechecks the verified administrator
under the same lock used for staff revocations and rejects stale revisions.
No player, match, tournament, badge, or independent leaderboard data is changed.

The migration was rehearsed on staging. The transaction checks cover draft/live
isolation, save/publish/restore/unpublish/republish, stale and revoked identities,
audit completeness, client-role permissions, and leaderboard preservation;
all rehearsal test data was rolled back.

The web build waits for the new API capability before replacing the current
website. The existing no-send SMTP preflight has a two-minute timeout so a
connection stall fails safely instead of occupying the deployment queue for
hours. It remains a required check.

## Verification

- API contract and production deployment checks.
- Web component and TypeScript checks, including prior production regressions.
- Desktop and mobile browser checks for visible Save/Publish controls,
  persistence after reload, separate leaderboard/website publication, and
  website tabs.
- Production publication is complete only after the API and web deployment
  attest the release and live public pages are checked.

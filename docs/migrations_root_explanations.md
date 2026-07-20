# Root migration explanations (legacy)

This file is an allowlist for **new** SQL files added under `migrations/`.

Policy:

- `supabase/migrations/` is canonical for Supabase schema migrations.
- New `migrations/*.sql` files should be rare and must include a justification below.

Entry format:

- `migrations/<filename>.sql`: <why this must live in root migrations and not only in supabase/migrations/>.

Current entries:

- `migrations/20260701_league_manager_end_wizard_columns.sql`: Legacy root migration used by the existing Streamlit League Manager end-of-league wizard path to add lifecycle, schedule, court-board, rules, and awards configuration columns on `leagues_metadata` before the Next/Vercel League Manager migration is complete.
- `migrations/20261019_tournament_registration_partner_links.sql`: Extends the existing root-level tournament registration schema with linked doubles partner request/team-link tables alongside the prior root registration migrations.
- `migrations/20261020_tournament_draw_team_uniqueness.sql`: Legacy root migration for the Streamlit/Next shared tournament ops schema; it relaxes the original tournament-wide `tournament_teams` team-number uniqueness so each division draw can independently use team slots 1..N while preserving legacy tournament-wide uniqueness for rows without `draw_id`.
- `migrations/20261020_tournament_registrations_player_id_postgrest_reload.sql`: Idempotent production hotfix for tournament registration submissions that need `tournament_registrations.player_id`; also sends `notify pgrst, 'reload schema'` so PostgREST refreshes its schema cache after DDL.
- `migrations/20261021_tournament_podium_draw_scope.sql`: Legacy root migration for the shared Streamlit/Next tournament ops schema; it adds `draw_id` to podium rows and replaces tournament-wide podium uniqueness with draw-scoped uniqueness while preserving legacy null-draw podium behavior.
- `migrations/20261022_tournament_matches_unique_publish.sql`: Legacy root migration for the shared Streamlit/Next tournament ops schema; it adds a partial unique index so each `tournament_games.id` can be published to official `matches` at most once while preserving non-tournament match history.
- `migrations/20261023_tournament_playoff_winner_bonus.sql`: Legacy root migration for the shared Streamlit/Next match schema; it adds optional match-level rating bonus provenance columns used when Tournament Ops publishes semifinal, bronze, or gold playoff winners with a configured winner-only Elo boost.
- `migrations/20261024_singles_rating_support.sql`: Legacy root migration for the shared Streamlit/Next match schema; it adds player singles-rating counters and lets official match rows be marked `match_format=singles` with null doubles partner columns for one-on-one tournament and single-match input.
- `migrations/20261025_public_partner_board_source.sql`: Legacy root migration for the shared public tournament partner board; it extends the partner-request source constraint to accept requests submitted from the Next public partner board before automatic acceptance creates confirmed team links.
- `migrations/20261026_public_support_requests.sql`: Legacy root migration for the shared public support/correction/privacy intake queue used by Next public forms while Streamlit remains the admin fallback and source of operational review.
- `migrations/20261027_league_live_sessions.sql`: Legacy root migration for the shared Streamlit/Next League Manager live workflow; it persists Next League Live sessions, per-round state, and court snapshots so operators can resume interrupted league nights while Streamlit remains the prototype/fallback.
- `migrations/20261028_public_support_intake_guardrails.sql`: Legacy-root mirror of the canonical service-role-only support intake fingerprint and privacy fulfillment columns; the canonical Supabase migration remains the source of truth.

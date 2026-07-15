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

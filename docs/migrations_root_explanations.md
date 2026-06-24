# Root migration explanations (legacy)

This file is an allowlist for **new** SQL files added under `migrations/`.

Policy:

- `supabase/migrations/` is canonical for Supabase schema migrations.
- New `migrations/*.sql` files should be rare and must include a justification below.

Entry format:

- `migrations/<filename>.sql`: <why this must live in root migrations and not only in supabase/migrations/>.

Current entries:

- `migrations/20261019_tournament_registration_partner_links.sql`: Extends the existing root-level tournament registration schema with linked doubles partner request/team-link tables alongside the prior root registration migrations.
- `migrations/20261020_tournament_registrations_player_id_postgrest_reload.sql`: Idempotent production hotfix for tournament registration submissions that need `tournament_registrations.player_id`; also sends `notify pgrst, 'reload schema'` so PostgREST refreshes its schema cache after DDL.

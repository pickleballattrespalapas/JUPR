# Root migration explanations (legacy)

This file is an allowlist for **new** SQL files added under `migrations/`.

Policy:

- `supabase/migrations/` is canonical for Supabase schema migrations.
- New `migrations/*.sql` files should be rare and must include a justification below.

Entry format:

- `migrations/<filename>.sql`: <why this must live in root migrations and not only in supabase/migrations/>.

Current entries:

- `migrations/20261019_tournament_registration_partner_links.sql`: Extends the existing root-level tournament registration schema with linked doubles partner request/team-link tables alongside the prior root registration migrations.

# Root migration explanations (legacy)

This file is an allowlist for **new** SQL files added under `migrations/`.

Policy:

- `supabase/migrations/` is canonical for Supabase schema migrations.
- New `migrations/*.sql` files should be rare and must include a justification below.

Entry format:

- `migrations/<filename>.sql`: <why this must live in root migrations and not only in supabase/migrations/>.

Current entries:

- _None yet._

# Supabase data API boundary

The Next application uses Supabase in two deliberately separate ways:

1. The browser uses the publishable/anonymous key only for Supabase Auth.
2. The browser sends the short-lived user JWT to FastAPI. FastAPI validates the
   user, club, role, and permission before using its server-only service-role
   client for Postgres reads and writes.

No JUPR business table, view, sequence, or RPC is a direct browser Data API.
Ratings, replay, badges, league movement, and tournament processing remain in
Python/FastAPI domain services.

## Migration contract

`20260719155515_server_only_data_api_lockdown.sql` is the canonical access
boundary. It:

- removes `public` schema usage and object privileges from `anon` and
  `authenticated`;
- enables RLS on every public base/partitioned table as defense in depth;
- explicitly retains service-role access to tables, views, sequences, and RPCs;
- changes default privileges so future Data API exposure must be intentional;
- reloads PostgREST's schema cache.

`20260719155737_canonicalize_server_only_tables.sql` backports the Weekly Recap,
support-intake, and League Live tables from legacy root migrations into the
canonical Supabase migration history, with the same server-only grants.

This matches Supabase's current two-layer model: grants decide whether a role can
reach an object, while RLS decides which rows are visible after access is granted.
An RLS-enabled server table intentionally has no browser policy because neither
browser role has an object grant.

## Staging verification

After review and staging apply, verify all three layers before deploying code:

1. Run the Supabase Security Advisor and review every new finding.
2. Confirm `anon` and `authenticated` have no public-schema usage, table CRUD,
   sequence access, or RPC execute permission.
3. Confirm `service_role` can perform the application-required operations.
4. Exercise public reads and authenticated admin writes through FastAPI.
5. Verify direct Data API requests made with publishable and authenticated user
   credentials fail without returning rows or private fields.

Do not weaken this boundary to fix a PostgREST permission error. Add an explicit
grant only when a reviewed product decision intentionally creates a direct Data
API surface, and pair that grant with least-privilege RLS policies and tests.

### Read-only baseline captured 2026-07-19

The connected `JUPR Staging` project was inspected without applying DDL:

- zero public tables/views had any `anon` or `authenticated` CRUD grant;
- zero public functions were executable by `anon` or `authenticated`;
- both browser roles lacked `USAGE` and `CREATE` on `public`;
- the named admin, replay, recap, subscription/outbox, support, live-session,
  and League Live tables had RLS enabled and explicit service-role CRUD access;
- the Security Advisor returned 61 informational `rls_enabled_no_policy`
  findings, expected for service-only RLS tables with no browser grants, and one
  unrelated warning to enable leaked-password protection in Auth settings.

The Auth warning is tracked with the advisor's current remediation guide:
[Password security](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection).
Re-run this evidence after the reviewed migration is applied and again before
production promotion; the dated baseline is not a substitute for those checks.

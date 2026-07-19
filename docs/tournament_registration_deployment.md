# Tournament registration deployment notes

## Secure returning registrant edit links

Streamlit and FastAPI deployments that enable returning registrants to request secure edit links must define a root-level secret or environment variable named `JUPR_REGISTRATION_EDIT_SECRET`. FastAPI also requires `SUPABASE_SERVICE_ROLE_KEY`; the public edit API fails closed rather than using the anon key or a signing-secret fallback.

Use a long random value containing at least 32 bytes of secret material. Do not reuse the SMTP password, Supabase keys, service-role keys, anon/public keys, or any other credential that may be shared with another service or exposed publicly.

Apply `supabase/migrations/20260719160821_public_registration_edit_transaction.sql` before deploying the matching FastAPI/Next build. The migration installs the service-role-only atomic edit RPC and backfills required selection write versions.

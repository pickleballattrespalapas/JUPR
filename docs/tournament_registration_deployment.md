# Tournament registration deployment notes

## Secure returning registrant edit links

Streamlit deployments that enable returning registrants to request secure edit links must define a root-level secret or environment variable named `JUPR_REGISTRATION_EDIT_SECRET`.

Use a long random value of at least 32 characters. Do not reuse the SMTP password, Supabase keys, service-role keys, anon/public keys, or any other credential that may be shared with another service or exposed publicly.

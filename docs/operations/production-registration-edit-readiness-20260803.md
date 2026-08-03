# Production registration-edit readiness — 2026-08-03

A read-only production diagnostic inspected the Fly application secret names and the GitHub `production` environment without exposing secret values.

## Result

- Production environment: `production`
- Production business-write policy: `enabled`
- Production tournament write gate: enabled
- `JUPR_REGISTRATION_EDIT_SECRET` configured on Fly: **no**
- `JUPR_REGISTRATION_CONFIRMATION_SECRET` configured on Fly: **no**
- Required SMTP configuration on Fly: **not configured**
- Required SMTP configuration in the GitHub `production` environment: **not configured**

Required SMTP values currently absent in both locations:

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM_EMAIL`

The public registration-edit endpoint should remain fail-closed until a stable signing secret, a tested transactional email sender, and shared/edge request throttling are in place.

The temporary diagnostic workflow and trigger were removed after the check.

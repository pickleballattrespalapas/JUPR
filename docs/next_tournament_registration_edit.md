# Next Tournament Registration Edit

This document tracks the tokenized public tournament-registration edit flow in Next.js and FastAPI.

## Scope in this slice

- FastAPI edit-link request endpoint: `POST /clubs/{club_slug}/tournament-registration/edit-link/request`.
- FastAPI edit verification endpoint: `GET /clubs/{club_slug}/tournament-registration/edit`.
- FastAPI edit submission endpoint: `POST /clubs/{club_slug}/tournament-registration/edit`.
- Next edit-link request form on `/clubs/[clubSlug]/tournament-registration`.
- Next edit route: `/clubs/[clubSlug]/tournament-registration/edit`.
- Existing HMAC edit-token builder/verifier from Python: `build_registration_edit_token` and `verify_registration_edit_token`.
- Existing registration persistence path from Python: `save_registration(..., expected_registration_id=...)`.
- Existing tournament registration edit email template/sender.
- Locked-email edit form: the email associated with the token is displayed but cannot be changed.
- Event/partner selections are resubmitted through the same Python validation path used by public registration.

## Data safety

The browser does not create or validate tokens locally and does not write to Supabase. It asks FastAPI to request an edit link, and FastAPI looks up the registration, builds the token, and sends the email server-side. The request response is intentionally non-enumerating: the browser receives the same generic acceptance message whether or not a matching registration exists.

The edit page is token-gated and can return private registration fields for that registration owner, such as phone and partner-contact fields, only after FastAPI validates tournament id, registration id, club scope, and email hash. Public roster and partner-board routes remain contact-safe.

## Runtime configuration

Edit tokens require the existing registration edit secret resolution:

```text
JUPR_REGISTRATION_EDIT_SECRET
```

The code can also derive a fallback signing secret from configured Supabase credentials, matching the existing Streamlit helper. Production should use an explicit stable secret.

Email delivery uses the existing email safety modes:

```text
JUPR_EMAIL_MODE=dry_run|staging_redirect|live
JUPR_STAGING_EMAIL_REDIRECT_TO=<safe inbox for staging_redirect>
```

Staging still defaults to dry-run unless live sending is explicitly allowed by the existing configuration guard.

## Explicitly out of scope

- Changing the registration email address from an edit link.
- Organizer moderation or admin registration management.
- Partner request accept/decline flows.
- Draw, team, game, podium, or tournament-operation writes.

## Follow-up slices

- Add organizer/admin registration management.
- Add public partner request flow with anti-abuse and moderation controls.
- Add operator-facing email delivery diagnostics if needed.

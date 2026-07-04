# Next Tournament Registration Edit

This document tracks the tokenized public tournament-registration edit flow in Next.js and FastAPI.

## Scope in this slice

- FastAPI edit verification endpoint: `GET /clubs/{club_slug}/tournament-registration/edit`.
- FastAPI edit submission endpoint: `POST /clubs/{club_slug}/tournament-registration/edit`.
- Next route: `/clubs/[clubSlug]/tournament-registration/edit`.
- Existing HMAC edit-token verifier from Python: `verify_registration_edit_token`.
- Existing registration persistence path from Python: `save_registration(..., expected_registration_id=...)`.
- Locked-email edit form: the email associated with the token is displayed but cannot be changed.
- Event/partner selections are resubmitted through the same Python validation path used by public registration.

## Data safety

The browser does not validate tokens locally and does not write to Supabase. It sends the edit token to FastAPI, and FastAPI validates tournament id, registration id, club scope, and email hash before returning or updating the registration.

The edit page is token-gated and can return private registration fields for that registration owner, such as phone and partner-contact fields, only after the token is verified. Public roster and partner-board routes remain contact-safe.

## Runtime configuration

Edit tokens require the existing registration edit secret resolution:

```text
JUPR_REGISTRATION_EDIT_SECRET
```

The code can also derive a fallback signing secret from configured Supabase credentials, matching the existing Streamlit helper. Production should use an explicit stable secret.

## Explicitly out of scope

- Generating or emailing edit links from Next.
- Changing the registration email address from an edit link.
- Organizer moderation or admin registration management.
- Partner request accept/decline flows.

## Follow-up slices

- Add safe edit-link request and email delivery flow.
- Add organizer/admin registration management.
- Add public partner request flow with anti-abuse and moderation controls.

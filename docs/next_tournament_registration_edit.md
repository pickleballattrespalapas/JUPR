# Next Tournament Registration Edit

This document tracks the tokenized public tournament-registration edit flow in Next.js and FastAPI.

## Scope in this slice

- FastAPI edit-link request endpoint: `POST /clubs/{club_slug}/tournament-registration/edit-link/request`.
- FastAPI edit verification endpoint: `GET /clubs/{club_slug}/tournament-registration/edit`.
- FastAPI edit submission endpoint: `POST /clubs/{club_slug}/tournament-registration/edit`.
- Next edit-link request form on `/clubs/[clubSlug]/tournament-registration`.
- Next edit route: `/clubs/[clubSlug]/tournament-registration/edit`.
- Existing HMAC edit-token builder/verifier from Python: `build_registration_edit_token` and `verify_registration_edit_token`.
- Atomic registration edit RPC invoked only by Python/FastAPI with the Supabase service role.
- Existing tournament registration edit email template/sender.
- Locked-email edit form: the email associated with the token is displayed but cannot be changed.
- Event/partner selections are resubmitted through the same Python validation path used by public registration.

## Data safety

The browser does not create or validate tokens locally and does not write to Supabase. It asks FastAPI to request an edit link, and FastAPI looks up the registration, builds the token, and sends the email server-side. The request response is intentionally non-enumerating: the browser receives the same generic acceptance message whether or not a matching registration exists.

The edit page is token-gated and can return private registration fields for that registration owner, such as phone and partner-contact fields, only after FastAPI validates tournament id, registration id, club scope, and email hash. Public roster and partner-board routes remain contact-safe.

The GET response is private and `no-store`. Both GET and POST refuse public editing once any selected event has been imported into a tournament draw. POST sends the registration version and every selection version through one compare-and-swap RPC. The RPC locks the registration scope, rechecks draw import and relationship state, updates retained selection IDs in place, and commits the registration plus selection diff as one transaction. A stale version, imported draw, relationship conflict, or database error leaves all rows unchanged.

After a successful commit, FastAPI attempts an updated confirmation email. The API returns only a safe delivery projection (`sent`, `staging_redirect`, `dry_run`, `failed`, or `unknown`) and never returns the recipient, provider message id, edit token, or signing material. Email failure does not falsely report the already-committed database edit as failed.

## Runtime configuration

Edit tokens require the existing registration edit secret resolution:

```text
JUPR_REGISTRATION_EDIT_SECRET
```

The legacy token helper can still derive a fallback signing secret from configured Supabase credentials for Streamlit compatibility. The public FastAPI edit routes intentionally reject that fallback: `JUPR_REGISTRATION_EDIT_SECRET` (or the equivalent private Streamlit secret) must be explicitly configured so Supabase key rotation cannot invalidate outstanding edit links.

The edit-link request, private edit GET, and edit POST also fail closed unless `SUPABASE_SERVICE_ROLE_KEY` is present on FastAPI. The browser never receives that credential, and the transactional edit function revokes execute access from `PUBLIC`, `anon`, and `authenticated` while granting only `service_role`.

Email delivery uses the existing email safety modes:

```text
JUPR_EMAIL_MODE=dry_run|staging_redirect|live
JUPR_STAGING_EMAIL_REDIRECT_TO=<safe inbox for staging_redirect>
```

Staging still defaults to dry-run unless live sending is explicitly allowed by the existing configuration guard.

## Abuse controls

Edit-link requests remain non-enumerating and honeypot-protected. This repository does not currently provide a shared, multi-instance request-throttling store or middleware. Apply rate limiting to `POST /clubs/*/tournament-registration/edit-link/request` at the Fly/Vercel edge or API gateway before enabling broad public traffic; an in-process limiter would not be reliable across Fly instances.

## Explicitly out of scope

- Changing the registration email address from an edit link.
- Organizer moderation or admin registration management.
- Partner request accept/decline flows.
- Draw, team, game, podium, or tournament-operation writes.

## Follow-up slices

- Add organizer/admin registration management.
- Add a shared durable API rate-limit backend if platform throttling is removed.
- Add operator-facing email delivery diagnostics if needed.

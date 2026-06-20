# Next.js → FastAPI Admin Auth Design (future implementation)

## Purpose

Define the minimum production-grade authentication and authorization contract required before enabling Next.js admin score entry.

This document is a design stub only. **No auth behavior is implemented by this document** and current runtime behavior must remain unchanged.

## Scope

In scope:
- Next.js admin clients calling FastAPI admin endpoints.
- Server-side token validation and role/permission checks.
- Club-scoped authorization for write operations.
- Audit attribution for admin write actions.

Out of scope:
- Implementing auth in this PR.
- Enabling Next.js admin score entry.
- Changing existing endpoint behavior.

## 1) Authentication source and transport

### Source of truth

- Authentication source is **Supabase Auth JWT**.
- User identity is represented by Supabase claims (user ID and email).

### Request flow

1. User signs in through approved Next.js auth flow.
2. Next.js obtains a Supabase Auth access token.
3. Next.js calls FastAPI admin endpoints with:
   - `Authorization: Bearer <supabase_access_token>`
4. FastAPI validates the bearer token before any admin write logic.

### FastAPI validation requirements

FastAPI must reject requests when token is:
- missing,
- malformed,
- invalid signature,
- expired,
- otherwise unverifiable.

## 2) Authorization model

After token validation, FastAPI must authorize the request as follows:

1. Resolve authenticated actor identity (`user_id`, `email`) from token claims.
2. Query `admin_role_assignments` for active role bindings.
3. Apply permissions using `jupr_app.domain.admin.roles` matrix.
4. Require `enter_scores` permission for score-entry routes.
5. Enforce `club_id` scope for all writes.

### Club-scoped write requirement

Every write endpoint must verify actor has permission for the target `club_id`.
If actor lacks assignment for that club, request must be denied.

## 3) Audit requirements

All admin write actions must write an audit record to `admin_activity_log` with at least:

- `actor_email`
- `actor_role`
- `source_page`
- `source_client`

Audit fields should be captured from request context and resolved role assignment.

## 4) Security requirements (non-negotiable)

- No static admin API token for user-initiated actions.
- No client-side Supabase service role key usage.
- No direct browser writes to Supabase rating tables.
- API must reject missing/invalid/expired tokens.
- API must reject users without club permission.

## 5) Rollout sequence

Rollout must proceed in this order:

1. **Read-only public Next.js** first.
2. **Authenticated scorekeeper login** second.
3. **Score entry enabled in staging only** third.
4. **Rated production writes** only after:
   - audit logging is verified, and
   - rollback path is tested.

## Implementation gate for future PRs

Before enabling Next.js admin score entry for rated workflows, future implementation PRs must satisfy this design contract end-to-end.


## Implementation status (Prompt 07)

- FastAPI admin score entry route now uses Bearer JWT auth scaffolding for Supabase tokens.
- Authorization resolves role from `admin_role_assignments` by `club_id` + `email/user_id` and enforces `enter_scores`.
- `read_only` and wrong-club bindings are denied with `403`; missing/invalid auth returns `401`.
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` is still disabled by default and blocks before auth/write work.
- Next.js admin score entry remains non-production-active by default.


## Audit runtime behavior (Prompt 08)

- `POST /admin/clubs/{club_id}/matches/batch` now attempts audit writes on successful writes and denied authenticated attempts.
- Denied authenticated attempts are logged with `flagged_for_review=true`.
- Unauthenticated requests are not audit-logged to avoid token leakage risk.
- Strict mode: set `JUPR_REQUIRE_API_AUDIT_LOG=1` to fail writes when audit persistence fails.
- Default mode degrades gracefully if `admin_activity_log` is unavailable in staging.

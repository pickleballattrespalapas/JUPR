#!/usr/bin/env python3
"""Dedicated, email-free QA identity for the three synthetic staging clubs.

Provisioning is explicit; ordinary runs never restore removed/revoked access.
Auth Admin creates a confirmed identity without a password or email delivery.
Each run gets a bounded session; the deploy workflow always ends that session.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

try:
    from scripts import prepare_parity_staging_session as auth
except ModuleNotFoundError:
    import prepare_parity_staging_session as auth

EMAIL = "pcs-staging-qa@example.invalid"
MARKER = "pcs-three-club-qa-v1"
CLUBS = ("cabo-test-club", "la-paz-test-club", "la-ribera-pickelball-club")
Error = auth.SessionPreparationError


def prepare(*, env, provision=False, transport=auth._default_transport, mask=lambda value: None):
    # Do this before any credential-bearing request or database mutation.
    if (env.get("GITHUB_ACTIONS") != "true"
            or env.get("GITHUB_REF") != "refs/heads/staging"
            or env.get("STAGING_SUPABASE_URL") != auth.EXPECTED_SUPABASE_ORIGIN
            or env.get("STAGING_API_BASE_URL") != auth.EXPECTED_API_ORIGIN):
        raise Error("QA setup requires canonical staging and the isolated staging services.")
    service_key = auth._required_env(env, "STAGING_SUPABASE_SERVICE_ROLE_KEY")
    anon_key = auth._required_env(env, "STAGING_SUPABASE_ANON_KEY")
    env_path = Path(auth._required_env(env, "GITHUB_ENV"))

    def request(method, path, payload=None):
        return auth._request_json_value(
            method=method, url=auth.EXPECTED_SUPABASE_ORIGIN + path,
            headers={**auth._service_headers(service_key), "Prefer": "return=representation"}, payload=payload,
            transport=transport, operation="Staging QA identity setup")

    def roles():
        query = urlencode({"select": "*", "email": f"eq.{EMAIL}"})
        rows = request("GET", "/rest/v1/admin_role_assignments?" + query)
        if not isinstance(rows, list):
            raise Error("Unable to inspect QA assignments.")
        return rows

    # Read only public fixture metadata, then look up the dedicated Auth user.
    clubs = request("GET", "/rest/v1/clubs?" + urlencode({
        "select": "id,is_active", "id": "in.(" + ",".join(CLUBS) + ")"}))
    if (not isinstance(clubs, list) or {c.get("id") for c in clubs} != set(CLUBS)
            or any(c.get("is_active") is not True for c in clubs)):
        raise Error("The three active synthetic staging clubs are required.")
    user = None
    for page in range(1, 101):
        listing = request("GET", f"/auth/v1/admin/users?page={page}&per_page=100")
        users = listing.get("users") if isinstance(listing, dict) else None
        if not isinstance(users, list):
            raise Error("Unable to inspect staging QA identity.")
        matches = [u for u in users if str(u.get("email", "")).lower() == EMAIL]
        if matches:
            if len(matches) != 1:
                raise Error("Staging QA identity is ambiguous.")
            user = matches[0]
            break
        if len(users) < 100:
            break
    else:
        raise Error("Staging identity lookup exceeded its bounded page limit.")

    existing_roles = roles()
    if user is None:
        if not provision or existing_roles:
            raise Error("Dedicated QA identity is missing; explicit provisioning is required.")
        user = request("POST", "/auth/v1/admin/users", {
            "email": EMAIL, "email_confirm": True,
            "app_metadata": {"pcs_staging_qa": MARKER},
            "user_metadata": {"display_name": "PCS staging QA"}})
    if (not isinstance(user, dict)
            or user.get("app_metadata", {}).get("pcs_staging_qa") != MARKER
            or user.get("banned_until") or user.get("deleted_at")):
        raise Error("Refusing to reuse an unrelated, banned, or deleted Auth identity.")
    user_id = auth._validated_uuid(user.get("id"), operation="Staging QA identity")
    identity = auth._AdminAssignment("qa", user_id, EMAIL, "administrator")
    auth._validate_auth_admin_user(user, identity)

    for row in existing_roles:
        if (row.get("club_id") not in CLUBS or row.get("role") != "administrator"
                or row.get("revoked_at") or row.get("expires_at")
                or row.get("user_id") not in (None, user_id)):
            raise Error("QA assignments have changed; refusing to restore or expand access.")
    present = {r["club_id"] for r in existing_roles}
    missing = set(CLUBS) - present
    if missing and not provision:
        raise Error("QA assignments are incomplete; ordinary tests never grant access.")
    if missing:
        # The existing, verified fixture administrator is the audited grantor.
        rows = request("GET", "/rest/v1/admin_role_assignments?" + urlencode({
            "select": "club_id,email,user_id", "club_id": "in.(" + ",".join(CLUBS) + ")",
            "role": "in.(administrator,club_owner)", "user_id": "not.is.null",
            "email": f"neq.{EMAIL}", "revoked_at": "is.null", "expires_at": "is.null"}))
        candidates = sorted({(r["user_id"], r["email"]) for r in rows
                             if all(any(s["club_id"] == c and s["user_id"] == r["user_id"]
                                        and s["email"] == r["email"] for s in rows) for c in CLUBS)})
        if not candidates:
            raise Error("A verified administrator for all three fixture clubs is required.")
        actor_id, actor_email = candidates[0]
        actor_id = auth._validated_uuid(actor_id, operation="Staging fixture administrator")
        actor = request("GET", f"/auth/v1/admin/users/{actor_id}")
        auth._validate_auth_admin_user(actor, auth._AdminAssignment("actor", actor_id, actor_email, "administrator"))
        for club in sorted(missing):
            request("POST", "/rest/v1/rpc/pcs_save_staff", {
                "p_club_id": club, "p_actor_email": actor_email, "p_actor_id": actor_id,
                "p_email": EMAIL, "p_role": "administrator", "p_scopes": [],
                "p_expires_at": None, "p_revoke": False})

    # Bind only this confirmed identity's three existing unbound assignments.
    for row in roles():
        if row.get("user_id") is None and row.get("club_id") in CLUBS:
            request("PATCH", "/rest/v1/admin_role_assignments?" + urlencode({
                "club_id": f"eq.{row['club_id']}", "email": f"eq.{EMAIL}",
                "user_id": "is.null", "revoked_at": "is.null"}), {"user_id": user_id})
    final_roles = roles()
    if (len(final_roles) != len(CLUBS) or {r.get("club_id") for r in final_roles} != set(CLUBS)
            or any(r.get("user_id") != user_id or r.get("role") != "administrator"
                   or r.get("revoked_at") or r.get("expires_at") for r in final_roles)):
        raise Error("QA assignments are not bound exclusively to the three test clubs.")

    generated = request("POST", "/auth/v1/admin/generate_link", {"type": "magiclink", "email": EMAIL})
    token_hash, verification_type = auth._validate_generate_link(generated, identity)
    payload = auth._request_json(
        method="POST", url=auth.EXPECTED_SUPABASE_ORIGIN + "/auth/v1/verify",
        headers=auth._anon_headers(anon_key), payload={"token_hash": token_hash, "type": verification_type},
        transport=transport, operation="Staging QA session verification")
    candidate = auth._candidate_access_token(payload)
    try:
        if candidate:
            mask(candidate)
            auth._append_github_env(env_path, {"STAGING_ADMIN_BEARER_TOKEN": candidate})
        token = auth._validate_access_token(payload, identity)
        workspaces = auth._request_json(
            method="GET", url=auth.EXPECTED_API_ORIGIN + "/admin/auth/workspaces",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            transport=transport, operation="Staging QA workspace verification")
        workspaces = workspaces.get("workspaces", [])
        if (len(workspaces) != len(CLUBS) or {w.get("club_id") for w in workspaces} != set(CLUBS)
                or any(w.get("roles") != ["administrator"] for w in workspaces)):
            raise Error("The live API did not grant exactly the three QA workspaces.")
        auth._append_github_env(env_path, {"STAGING_ADMIN_EMAIL": EMAIL})
    except Error:
        if candidate:
            auth.cleanup_staging_session(env={**env, "STAGING_ADMIN_BEARER_TOKEN": candidate}, transport=transport)
        raise
    return {"status": "passed", "account": EMAIL, "club_ids": list(CLUBS),
            "role": "administrator", "email_sent": False, "password_created": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provision", action="store_true")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = prepare(env=os.environ, provision=args.provision,
                         mask=lambda value: print(f"::add-mask::{value}"))
        auth._write_report(args.report_dir, "qa-account.json", report)
    except Error as exc:
        print(f"Staging QA setup failed: {exc}", file=sys.stderr)
        return 1
    print("Dedicated staging QA session ready for three synthetic clubs; no email sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

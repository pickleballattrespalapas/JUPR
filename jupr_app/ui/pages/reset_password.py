from __future__ import annotations

import logging

import streamlit as st
import streamlit.components.v1 as components

from jupr_app.ui.admin_auth import (
    AdminAuthConfigError,
    AdminAuthError,
    clear_local_admin_auth_state,
    establish_recovery_session,
    get_recovery_query_params,
    is_recovery_flow_query,
    make_supabase_auth_client,
    update_recovered_user_password,
)
from jupr_app.ui.layout import page_shell


_MIN_PASSWORD_LENGTH = 8
logger = logging.getLogger(__name__)


def _query_param_text(key: str) -> str:
    value = st.query_params.get(key, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _inject_hash_to_query_probe_bridge() -> None:
    """
    Supabase recovery links can return tokens in URL hash fragment.
    Streamlit server code can't read hash values, so this probe marks when
    hash inspection happened and moves recovery fields into query params once.
    """
    components.html(
        """
        <script>
        (function () {
          const appWindow = window.parent || window;
          const currentUrl = new URL(appWindow.location.href);
          const recoveryKeys = ["access_token", "refresh_token", "type", "code", "token_hash"];
          const hasRecoveryQuery = recoveryKeys.some((k) => !!currentUrl.searchParams.get(k));
          const probeReady = currentUrl.searchParams.get("_recovery_probe") === "1";
          if (hasRecoveryQuery || probeReady) return;

          const hash = appWindow.location.hash || "";
          const hashParams = hash && hash.length > 1
            ? new URLSearchParams(hash.substring(1))
            : new URLSearchParams();

          const hasRecoveryHash = recoveryKeys.some((k) => !!hashParams.get(k));

          if (hasRecoveryHash) {
            recoveryKeys.forEach((key) => {
              const value = hashParams.get(key);
              if (value && !currentUrl.searchParams.get(key)) {
                currentUrl.searchParams.set(key, value);
              }
            });
          }

          currentUrl.searchParams.set("_recovery_probe", "1");
          currentUrl.hash = "";

          appWindow.location.replace(currentUrl.toString());
        })();
        </script>
        """,
        height=0,
    )


def _should_wait_for_probe(has_recovery_query: bool, recovery_probe: str) -> bool:
    return (not has_recovery_query) and recovery_probe != "1"


def render(ctx):
    page_shell(
        "🔐 Reset Password",
        "Set a new password for your admin account.",
        mode_label="Public",
    )

    try:
        client = make_supabase_auth_client()
    except AdminAuthConfigError as exc:
        st.error(str(exc))
        st.stop()

    recovery_params = get_recovery_query_params()
    recovery_probe = _query_param_text("_recovery_probe")
    has_recovery_query = is_recovery_flow_query(recovery_params)
    recovery_keys = sorted(recovery_params.keys())

    if recovery_keys:
        logger.info("Recovery params detected keys only: %s", recovery_keys)
    logger.info(
        "Reset page probe state: has_recovery_query=%s _recovery_probe=%s",
        has_recovery_query,
        recovery_probe or "<empty>",
    )

    if _should_wait_for_probe(has_recovery_query, recovery_probe):
        _inject_hash_to_query_probe_bridge()
        st.info("Preparing reset link...")
        st.stop()

    recovery_session_ready = establish_recovery_session(client, recovery_params)
    if recovery_session_ready:
        logger.info("Recovery session established on reset page")

    if (not has_recovery_query) and recovery_probe == "1":
        st.error(
            "This password reset link is invalid or expired. Request a new reset email."
        )
        error_hint = recovery_params.get("error_description") or recovery_params.get("error")
        if error_hint:
            st.caption(f"Supabase returned: {error_hint}")
        st.stop()
    if not recovery_session_ready:
        st.error(
            "This password reset link is invalid or expired. Request a new reset email."
        )
        error_hint = recovery_params.get("error_description") or recovery_params.get("error")
        if error_hint:
            st.caption(f"Supabase returned: {error_hint}")
        st.stop()

    with st.form("reset_password_form"):
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Save new password")

    if submitted:
        clean_password = str(new_password or "")
        clean_confirm = str(confirm_password or "")

        if not clean_password or not clean_confirm:
            st.error("Enter and confirm your new password.")
            return

        if clean_password != clean_confirm:
            st.error("Passwords do not match.")
            return

        if len(clean_password) < _MIN_PASSWORD_LENGTH:
            st.error(f"Password must be at least {_MIN_PASSWORD_LENGTH} characters.")
            return

        try:
            update_recovered_user_password(client, clean_password)
        except AdminAuthError as exc:
            st.error(str(exc))
            return

        clear_local_admin_auth_state()
        st.success("Password updated successfully. Return to login and sign in with your new password.")
        st.link_button("Return to login", "?page=leaderboards")

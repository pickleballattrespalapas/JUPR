from __future__ import annotations

import streamlit as st

from jupr_app.ui.admin_auth import (
    AdminAuthConfigError,
    AdminAuthError,
    is_allowed_admin_email,
    login_admin,
    logout_admin,
    send_password_reset_email,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.page_registry import ADMIN_ONLY_PAGE_KEYS

DEFAULT_ADMIN_PAGE_KEY = "league_manager"
RESET_PASSWORD_REDIRECT_URL = "https://juprtrespalapas.streamlit.app/?page=reset_password&public=1"


def _resolve_post_login_target() -> str:
    requested_key = str(st.session_state.pop("post_login_admin_page_key", "") or "").strip().lower()
    if requested_key in ADMIN_ONLY_PAGE_KEYS:
        return requested_key

    requested_query_page = str(st.query_params.get("next", "") or "").strip().lower()
    if requested_query_page in ADMIN_ONLY_PAGE_KEYS:
        return requested_query_page

    return DEFAULT_ADMIN_PAGE_KEY


def render(ctx):
    page_shell(
        "🔐 Admin Login",
        "Authorized Tres Palapas staff only. Sign in to access JUPR administration.",
        mode_label="Public",
        right_html="",
    )

    st.markdown("<div style='max-width:560px;margin:1.5rem auto 0 auto;'>", unsafe_allow_html=True)

    st.markdown("### Sign in")

    with st.form("admin_login_page_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="name@trespalapas.com")
        password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if login_submitted:
        try:
            result = login_admin(email=email, password=password)
            login_user = result.get("user")
            login_email = str(getattr(login_user, "email", "") or "").strip().lower()
            admin_allowlist = st.session_state.get("admin_allowlist", set())
            if not is_allowed_admin_email(login_email, admin_allowlist):
                logout_admin()
                st.error("Authenticated but not authorized for admin access.")
            else:
                st.query_params["admin"] = "1"
                st.query_params.pop("public", None)
                st.query_params["page"] = _resolve_post_login_target()
                st.rerun()
        except AdminAuthConfigError as exc:
            st.error(str(exc))
        except AdminAuthError as exc:
            st.error(str(exc))

    with st.expander("Forgot password?"):
        with st.form("admin_login_page_reset_form", clear_on_submit=False):
            reset_email = st.text_input("Email for reset link")
            send_reset_submitted = st.form_submit_button("Send reset email", use_container_width=True)

        if send_reset_submitted:
            try:
                send_password_reset_email(
                    reset_email,
                    redirect_to=RESET_PASSWORD_REDIRECT_URL,
                )
            except AdminAuthConfigError as exc:
                st.error(str(exc))
            except AdminAuthError as exc:
                st.error(str(exc))
            else:
                st.success("If that email exists, a reset link has been sent.")

    st.markdown("</div>", unsafe_allow_html=True)

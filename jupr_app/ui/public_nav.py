from __future__ import annotations

import inspect

import streamlit as st

from jupr_app.ui.page_registry import LABEL_TO_PAGE_KEY, PAGE_KEY_TO_LABEL
from jupr_app.ui.public_links import navigate_same_tab

PUBLIC_LABEL_BY_KEY: dict[str, str] = {
    "home": "Home",
    "leaderboards": "Leaderboards",
    "players": "Players",
    "tournament_registration": "Events",
    "weekly_recap": "Weekly Recap",
    "rating_rules": "Rating Rules",
    "faqs": "FAQs",
}

PUBLIC_SOURCE_BY_PAGE: dict[str, str] = {
    "home": "public_header:home",
    "leaderboards": "public_header:leaderboards",
    "players": "public_header:players",
    "tournament_registration": "public_header:tournament_registration",
    "rating_rules": "public_header:rating_rules",
    "weekly_recap": "public_header:weekly_recap",
    "faqs": "public_header:faqs",
}

_SAFE_CONTEXT_KEYS = {"club_id"}


def _current_query_params() -> dict[str, str]:
    params: dict[str, str] = {}
    for key in st.query_params.keys():
        value = st.query_params.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        value_text = str(value or "").strip()
        if value_text:
            params[str(key)] = value_text
    return params


def _safe_context_params(page_key: str, current_params: dict[str, str]) -> dict[str, str]:
    params = {key: current_params[key] for key in _SAFE_CONTEXT_KEYS if current_params.get(key)}
    if page_key == "leaderboards" and current_params.get("league"):
        params["league"] = current_params["league"]
    return params


def _render_public_nav_styles() -> None:
    st.markdown(
        """
        <style>
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-nav {
            border: 1px solid color-mix(in srgb, var(--border-strong) 58%, transparent);
            border-radius: 14px;
            padding: 0.65rem 0.85rem;
            background: color-mix(in srgb, var(--panel) 95%, transparent);
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-nav-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.75rem;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand {
            flex: 0 1 auto;
            min-width: 10rem;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand .stButton > button {
            all: unset;
            cursor: pointer;
            display: block;
            font-size: 1.03rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            line-height: 1.1;
            color: var(--text-primary);
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand .stButton > button:hover {
            text-decoration: underline;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand-title {
            font-size: 1.03rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            line-height: 1.1;
            color: var(--text-primary);
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand-subtitle {
            margin-top: 0.1rem;
            font-size: 0.74rem;
            font-weight: 600;
            color: color-mix(in srgb, var(--text-muted) 95%, var(--text-secondary));
            white-space: nowrap;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-nav-control {
            flex: 1 1 34rem;
            min-width: min(32rem, 100%);
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-nav-control [role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin {
            flex: 0 1 auto;
            margin-left: auto;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin .stButton {
            display: inline-flex;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin .stButton > button {
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 800;
            padding: 0.35rem 0.65rem;
            white-space: nowrap;
            width: fit-content;
          }
          @media (max-width: 1080px) {
            .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin {
              margin-left: 0;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _single_select_widget(
    *,
    options: list[str],
    current_option: str,
    key: str,
) -> str:
    if hasattr(st, "pills"):
        pills_sig = inspect.signature(st.pills)
        kwargs: dict[str, object] = {"key": key}
        if "selection_mode" in pills_sig.parameters:
            kwargs["selection_mode"] = "single"
        if "default" in pills_sig.parameters:
            kwargs["default"] = current_option
        return st.pills("", options=options, **kwargs) or current_option

    if hasattr(st, "segmented_control"):
        segmented_sig = inspect.signature(st.segmented_control)
        kwargs = {"key": key}
        if "default" in segmented_sig.parameters:
            kwargs["default"] = current_option
        return st.segmented_control("", options=options, **kwargs) or current_option

    return st.selectbox("", options=options, index=options.index(current_option), key=key, label_visibility="collapsed")


def render_public_app_header(
    *,
    labels_in_order: list[str] | None = None,
    current_label: str | None = None,
    default_page: str = "home",
) -> str:
    """Render the public website-style top nav and return the active page label."""

    labels_in_order = labels_in_order or []
    nav_page_keys: list[str] = []
    for label in labels_in_order:
        page_key = LABEL_TO_PAGE_KEY.get(label)
        if not page_key:
            continue
        if page_key not in PUBLIC_LABEL_BY_KEY:
            continue
        if page_key not in nav_page_keys:
            nav_page_keys.append(page_key)

    if not nav_page_keys:
        nav_page_keys = list(PUBLIC_LABEL_BY_KEY.keys())

    current_page_key = LABEL_TO_PAGE_KEY.get(current_label or "")
    if current_page_key not in nav_page_keys:
        current_page_key = default_page if default_page in nav_page_keys else nav_page_keys[0]

    current_params = _current_query_params()
    _render_public_nav_styles()

    st.markdown(
        """
        <div class="jupr-public-shell jupr-public-nav-streamlit">
          <div class="jupr-public-nav" role="navigation" aria-label="Public site navigation">
            <div class="jupr-public-nav-toolbar">
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="jupr-public-brand">', unsafe_allow_html=True)
    if st.button("JUPR", key="public_header_brand_home"):
        navigate_same_tab(
            page="home",
            params=_safe_context_params("home", current_params),
            public_mode=True,
            source="public_header:home",
        )
    st.markdown('<div class="jupr-public-brand-subtitle">Tres Palapas Rating System</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="jupr-public-nav-control">', unsafe_allow_html=True)
    nav_labels = [PUBLIC_LABEL_BY_KEY[key] for key in nav_page_keys]
    current_label_value = PUBLIC_LABEL_BY_KEY[current_page_key]
    selected_label = _single_select_widget(
        options=nav_labels,
        current_option=current_label_value,
        key="public_header_nav_selector",
    )
    selected_page_key = next((k for k in nav_page_keys if PUBLIC_LABEL_BY_KEY[k] == selected_label), current_page_key)
    if selected_page_key != current_page_key:
        navigate_same_tab(
            page=selected_page_key,
            params=_safe_context_params(selected_page_key, current_params),
            public_mode=True,
            source=PUBLIC_SOURCE_BY_PAGE[selected_page_key],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="jupr-public-admin">', unsafe_allow_html=True)
    admin_authenticated = bool(st.session_state.get("jupr_admin_authenticated", False))
    admin_label = "Admin Dashboard" if admin_authenticated else "Admin Login"
    if st.button(admin_label, key="public_header_admin_primary"):
        navigate_same_tab(
            page="league_manager" if admin_authenticated else "admin_login",
            public_mode=False,
            source="public_header:admin_dashboard" if admin_authenticated else "public_header:admin_login",
        )

    if admin_authenticated and st.button("Logout", key="public_header_logout"):
        navigate_same_tab(
            page=current_page_key or default_page,
            params={"logout": "1", **_safe_context_params(current_page_key or default_page, current_params)},
            public_mode=True,
            source="public_header:logout",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    return PAGE_KEY_TO_LABEL.get(current_page_key, PAGE_KEY_TO_LABEL[default_page])


def render_public_top_nav(
    *,
    labels_in_order: list[str] | None = None,
    current_label: str | None = None,
    default_page: str = "home",
) -> str:
    """Backward-compatible alias for the public app header renderer."""
    return render_public_app_header(
        labels_in_order=labels_in_order,
        current_label=current_label,
        default_page=default_page,
    )

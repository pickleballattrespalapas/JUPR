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
          .jupr-public-nav-streamlit .jupr-public-nav-toolbar {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
          }
          .jupr-public-nav-streamlit .jupr-public-brand-block {
            min-width: max-content;
          }
          .jupr-public-nav-streamlit .jupr-public-nav-pills,
          .jupr-public-nav-streamlit .jupr-public-admin-pills {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
          }
          .jupr-public-nav-streamlit .jupr-public-nav-pills {
            flex: 1 1 560px;
          }
          .jupr-public-nav-streamlit .jupr-public-admin-pills {
            justify-content: flex-end;
            flex: 0 1 auto;
          }
          .jupr-public-nav-streamlit .stButton > button {
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--border-strong) 72%, transparent);
            padding: 0.34rem 0.72rem;
            color: var(--text-primary);
            background: color-mix(in srgb, var(--panel) 86%, transparent);
            text-decoration: none;
            font-size: 0.84rem;
            font-weight: 700;
            white-space: nowrap;
            min-height: 0;
            line-height: 1.15;
            width: fit-content;
            max-width: 100%;
          }
          .jupr-public-nav-streamlit .stButton > button:hover {
            border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
            background: var(--accent-soft);
            color: var(--text-primary);
          }
          .jupr-public-nav-streamlit .stButton > button:focus-visible {
            outline: 3px solid var(--focus);
            outline-offset: 2px;
          }
          .jupr-public-nav-streamlit .jupr-public-nav-active .stButton > button {
            border-color: color-mix(in srgb, var(--accent) 52%, var(--border));
            background: color-mix(in srgb, var(--accent-soft) 85%, var(--panel));
            box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 20%, transparent);
          }
          .jupr-public-nav-streamlit .jupr-public-brand-button .stButton > button {
            border-radius: 10px;
            padding: 0.44rem 0.7rem;
            font-weight: 800;
            text-align: left;
          }
          .jupr-public-nav-streamlit .jupr-public-admin-button .stButton > button {
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 800;
          }
          @media (max-width: 1080px) {
            .jupr-public-nav-streamlit .jupr-public-admin-pills {
              justify-content: flex-start;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_nav_button(
    *,
    page_key: str,
    label: str,
    active: bool,
    current_params: dict[str, str],
    position: int,
) -> None:
    active_class = " jupr-public-nav-active" if active else ""
    st.markdown(f'<div class="jupr-public-nav-item{active_class}">', unsafe_allow_html=True)
    if st.button(label, key=f"public_header_nav_{page_key}_{position}"):
        navigate_same_tab(
            page=page_key,
            params=_safe_context_params(page_key, current_params),
            public_mode=True,
            source=PUBLIC_SOURCE_BY_PAGE[page_key],
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _horizontal_container(parent: st.delta_generator.DeltaGenerator) -> st.delta_generator.DeltaGenerator:
    container_sig = inspect.signature(parent.container)
    if "horizontal" in container_sig.parameters:
        return parent.container(horizontal=True)
    return parent.container()


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
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="jupr-public-nav-toolbar">', unsafe_allow_html=True)
    with _horizontal_container(st):
        st.markdown('<div class="jupr-public-brand-block jupr-public-brand-button">', unsafe_allow_html=True)
        st.markdown('<div class="jupr-public-brand-button">', unsafe_allow_html=True)
        if st.button("JUPR\nTres Palapas Rating System", key="public_header_brand_home"):
            navigate_same_tab(
                page="home",
                params=_safe_context_params("home", current_params),
                public_mode=True,
                source="public_header:home",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="jupr-public-nav-pills">', unsafe_allow_html=True)
        for idx, page_key in enumerate(nav_page_keys):
            _render_nav_button(
                page_key=page_key,
                label=PUBLIC_LABEL_BY_KEY[page_key],
                active=page_key == current_page_key,
                current_params=current_params,
                position=idx,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="jupr-public-admin-pills">', unsafe_allow_html=True)
        admin_authenticated = bool(st.session_state.get("jupr_admin_authenticated", False))
        admin_label = "Admin Dashboard" if admin_authenticated else "Admin Login"
        st.markdown('<div class="jupr-public-admin-button">', unsafe_allow_html=True)
        if st.button(admin_label, key="public_header_admin_primary"):
            navigate_same_tab(
                page="league_manager" if admin_authenticated else "admin_login",
                public_mode=False,
                source="public_header:admin_dashboard" if admin_authenticated else "public_header:admin_login",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if admin_authenticated:
            st.markdown('<div class="jupr-public-admin-button">', unsafe_allow_html=True)
            if st.button("Logout", key="public_header_logout"):
                navigate_same_tab(
                    page=current_page_key or default_page,
                    params={"logout": "1", **_safe_context_params(current_page_key or default_page, current_params)},
                    public_mode=True,
                    source="public_header:logout",
                )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

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

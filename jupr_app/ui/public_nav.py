from __future__ import annotations

import inspect

import streamlit as st

from jupr_app.ui.branding import CLUB_NAME, PRODUCT_NAME, TAGLINE
from jupr_app.ui.page_registry import LABEL_TO_PAGE_KEY, PAGE_KEY_TO_LABEL
from jupr_app.ui.public_links import navigate_same_tab

PUBLIC_LABEL_BY_KEY: dict[str, str] = {
    "home": "Home",
    "leaderboards": "Leaderboards",
    "league_results": "League Results",
    "match_explorer": "Match Explorer",
    "players": "Players",
    "badge_codex": "Badges",
    "challenge_ladder": "Challenge Ladder",
    "jupr_live": "JUPR Live",
    "tournament_registration": "Tournaments",
    "tournament_partner_board": "Partner Board",
    "rating_rules": "Rating Rules",
    "weekly_recap": "Weekly Recap",
    "faqs": "FAQs",
}

PUBLIC_SOURCE_BY_PAGE: dict[str, str] = {
    "home": "public_header:home",
    "leaderboards": "public_header:leaderboards",
    "league_results": "public_header:league_results",
    "match_explorer": "public_header:match_explorer",
    "players": "public_header:players",
    "badge_codex": "public_header:badge_codex",
    "challenge_ladder": "public_header:challenge_ladder",
    "jupr_live": "public_header:jupr_live",
    "tournament_registration": "public_header:tournament_registration",
    "tournament_partner_board": "public_header:tournament_partner_board",
    "rating_rules": "public_header:rating_rules",
    "weekly_recap": "public_header:weekly_recap",
    "faqs": "public_header:faqs",
}

_SAFE_CONTEXT_KEYS = {"club_id"}
PUBLIC_FOOTER_LINKS: tuple[tuple[str, str], ...] = (
    ("Privacy", "privacy_policy"),
    ("Terms", "terms_of_use"),
    ("Contact", "contact_support"),
    ("Data Corrections", "data_corrections"),
    ("Email Preferences", "email_preferences"),
)


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
            min-width: 14rem;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-brand-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-logo-slot {
            width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--border-strong) 55%, transparent);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--text-muted);
            background: color-mix(in srgb, var(--panel) 88%, transparent);
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
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin-link {
            flex: 0 0 auto;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-admin-link .stButton > button {
            all: unset;
            cursor: pointer;
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 600;
            text-decoration: underline;
            white-space: nowrap;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-nav-control [role="radiogroup"] {
            gap: 0.35rem;
            flex-wrap: wrap;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-footer {
            margin-top: 1.2rem;
            padding-top: 0.65rem;
            border-top: 1px solid color-mix(in srgb, var(--border-strong) 38%, transparent);
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-footer-links {
            font-size: 0.76rem;
            color: var(--text-muted);
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-wrap: wrap;
            justify-content: center;
          }
          .jupr-public-shell.jupr-public-nav-streamlit .jupr-public-footer-links .stButton > button {
            all: unset;
            cursor: pointer;
            color: var(--text-muted);
            text-decoration: underline;
            font-size: 0.76rem;
            font-weight: 600;
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

    st.markdown('<div class="jupr-public-brand"><div class="jupr-public-brand-row"><div class="jupr-public-logo-slot" aria-hidden="true">Logo</div>', unsafe_allow_html=True)
    if st.button(f"{PRODUCT_NAME} at {CLUB_NAME}", key="public_header_brand_home"):
        navigate_same_tab(
            page="home",
            params=_safe_context_params("home", current_params),
            public_mode=True,
            source="public_header:home",
        )
    st.markdown('</div><div class="jupr-public-brand-subtitle">' + TAGLINE + '</div></div>', unsafe_allow_html=True)

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

    admin_authenticated = bool(st.session_state.get("jupr_admin_authenticated", False))
    admin_label = "Admin Dashboard" if admin_authenticated else "Admin Login"
    st.markdown('<div class="jupr-public-admin-link">', unsafe_allow_html=True)
    if st.button(admin_label, key="public_header_admin_link"):
        navigate_same_tab(
            page="league_manager" if admin_authenticated else "admin_login",
            public_mode=False,
            source="public_header:admin_dashboard" if admin_authenticated else "public_header:admin_login",
        )
    st.markdown("</div></div></div>", unsafe_allow_html=True)

    return PAGE_KEY_TO_LABEL.get(current_page_key, PAGE_KEY_TO_LABEL[default_page])


def render_public_footer(*, current_label: str | None = None, default_page: str = "home") -> None:
    current_params = _current_query_params()
    current_page_key = LABEL_TO_PAGE_KEY.get(current_label or "") or default_page
    admin_authenticated = bool(st.session_state.get("jupr_admin_authenticated", False))

    st.markdown('<div class="jupr-public-shell jupr-public-nav-streamlit"><div class="jupr-public-footer"><div class="jupr-public-footer-links">', unsafe_allow_html=True)
    for link_label, page_key in PUBLIC_FOOTER_LINKS:
        if st.button(link_label, key=f"public_footer_link_{page_key}"):
            navigate_same_tab(
                page=page_key,
                params=_safe_context_params(page_key, current_params),
                public_mode=True,
                source=f"public_footer:{page_key}",
            )

    if admin_authenticated and st.button("Logout", key="public_footer_logout"):
        navigate_same_tab(
            page=current_page_key or default_page,
            params={"logout": "1", **_safe_context_params(current_page_key or default_page, current_params)},
            public_mode=True,
            source="public_footer:logout",
        )
    st.markdown("</div></div></div>", unsafe_allow_html=True)


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

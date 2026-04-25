from __future__ import annotations

from urllib.parse import urlencode

import streamlit as st

from jupr_app.ui.page_registry import LABEL_TO_PAGE_KEY, PAGE_KEY_TO_LABEL

PUBLIC_LABEL_BY_KEY: dict[str, str] = {
    "home": "Home",
    "leaderboards": "Leaderboards",
    "players": "Players",
    "tournament_registration": "Events",
    "weekly_recap": "Weekly Recap",
    "rating_rules": "Rating Rules",
    "faqs": "FAQs",
}


def _current_query_params() -> dict[str, str]:
    params: dict[str, str] = {}
    for key in st.query_params.keys():
        value = st.query_params.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        value_text = str(value or "")
        if value_text:
            params[key] = value_text
    return params


def _href_for_page(page_key: str, *, base_params: dict[str, str]) -> str:
    params = dict(base_params)
    params.pop("admin", None)
    params.pop("public", None)
    params.pop("next", None)
    if page_key == "home":
        params.pop("page", None)
    else:
        params["page"] = page_key

    if not params:
        return "./"

    return f"?{urlencode(sorted(params.items()))}"


def render_public_top_nav(
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

    base_params = _current_query_params()

    nav_links_html = "".join(
        (
            f'<a class="jupr-public-nav-link {"active" if page_key == current_page_key else ""}" '
            f'href="{_href_for_page(page_key, base_params=base_params)}">{PUBLIC_LABEL_BY_KEY[page_key]}</a>'
        )
        for page_key in nav_page_keys
    )

    admin_href = "?admin=1&page=admin_login"

    st.markdown(
        f"""
        <div class="jupr-public-shell">
          <div class="jupr-public-nav" role="navigation" aria-label="Public site navigation">
            <a class="jupr-public-brand" href="./">
              <span>JUPR</span>
              <small>Tres Palapas Rating System</small>
            </a>
            <nav class="jupr-public-nav-links" aria-label="Public navigation">
              {nav_links_html}
            </nav>
            <a class="jupr-public-admin-action" href="{admin_href}">Admin Login</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return PAGE_KEY_TO_LABEL.get(current_page_key, PAGE_KEY_TO_LABEL[default_page])

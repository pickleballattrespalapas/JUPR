from __future__ import annotations

import streamlit as st

from jupr_app.ui.page_registry import PAGE_KEY_TO_LABEL
from jupr_app.ui.url import qp_get


def render_public_top_nav(
    *,
    labels_in_order: list[str] | None = None,  # noqa: ARG001
    current_label: str | None = None,  # noqa: ARG001
    default_page: str = "home",
) -> str:
    """Render the public website-style top nav and return the active page label."""

    current_page = (qp_get("page", "") or "").strip().lower() or default_page
    if current_page not in PAGE_KEY_TO_LABEL:
        current_page = default_page

    links: list[tuple[str, str, str]] = [
        ("Home", "./", "home"),
        ("Leaderboards", "?page=leaderboards", "leaderboards"),
        ("Players", "?page=players", "players"),
        ("Events", "?page=tournament_registration", "tournament_registration"),
        ("Updates", "?page=verified_updates_request", "verified_updates_request"),
        ("Admin Login", "?admin=1&page=admin_login", "admin_login"),
    ]

    nav_links_html = "".join(
        (
            f'<a class="jupr-public-nav-link {"active" if key == current_page else ""}" '
            f'href="{href}">{label}</a>'
        )
        for label, href, key in links
    )

    st.markdown(
        f"""
        <div class="jupr-public-shell">
          <header class="jupr-public-nav">
            <a class="jupr-public-brand" href="./">
              <span>JUPR</span>
              <small>Tres Palapas Rating System</small>
            </a>
            <nav class="jupr-public-nav-links" aria-label="Public navigation">
              {nav_links_html}
            </nav>
          </header>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return PAGE_KEY_TO_LABEL.get(current_page, PAGE_KEY_TO_LABEL[default_page])

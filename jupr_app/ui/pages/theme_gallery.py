from __future__ import annotations

import html

import streamlit as st

from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_clean import card, callout, divider, kpi_card, pill, theme_override_css


def _apply_theme_override(mode: str) -> None:
    if mode == "Force Light":
        css = theme_override_css("light")
    elif mode == "Force Dark":
        css = theme_override_css("dark")
    else:
        css = ""

    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render(ctx):
    PUBLIC_MODE = bool(getattr(ctx, "public_mode", False))
    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("🎨 Theme QA", "Style gallery for contrast checks across tokens.", mode_label=mode_label)

    theme_mode = st.selectbox(
        "Theme override",
        ["System", "Force Light", "Force Dark"],
        help="Use this to quickly preview light/dark theme tokens without changing OS settings.",
        index=0,
    )
    _apply_theme_override(theme_mode)

    st.markdown(
        """
        <style>
        .qa-section {
            margin-top: 1rem;
        }
        .qa-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        .qa-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--panel);
            padding: 12px;
            box-shadow: var(--shadow);
        }
        .qa-tone {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9rem;
        }
        .qa-tone .value {
            font-weight: 650;
        }
        .qa-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--panel);
            color: var(--text-primary);
        }
        .qa-table th,
        .qa-table td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            font-size: 0.85rem;
        }
        .qa-table th {
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            background: var(--table-stripe);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Typography")
    st.markdown(
        "This is **primary body text** with a [link](https://example.com) and some `inline code`."
    )
    st.caption("Caption text uses the muted token for secondary emphasis.")

    divider()
    st.markdown("### Status & text tokens")
    st.markdown(
        """
        <div class="qa-grid">
          <div class="qa-card qa-tone"><span>Primary text</span><span class="value">Readable</span></div>
          <div class="qa-card qa-tone"><span>Secondary text</span><span class="value" style="color:var(--text-secondary);">Readable</span></div>
          <div class="qa-card qa-tone"><span>Muted text</span><span class="value" style="color:var(--text-muted);">Muted</span></div>
          <div class="qa-card qa-tone"><span>Success</span><span class="value" style="color:var(--status-success);">+12%</span></div>
          <div class="qa-card qa-tone"><span>Warning</span><span class="value" style="color:var(--status-warning);">Needs attention</span></div>
          <div class="qa-card qa-tone"><span>Danger</span><span class="value" style="color:var(--status-danger);">Decline</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    divider()
    st.markdown("### Cards, callouts, KPIs, and badges")
    card("<strong>Card title</strong><br/>Card body text uses primary text tokens.")
    callout("info", "Info callout", "This callout uses theme-aware text colors.")

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        kpi_card("Active Players", "128", delta="▲ +12 vs last week", icon="👥")
    with kpi_cols[1]:
        kpi_card("Matches", "52", delta="▼ -3 vs last week", icon="🏆")
    with kpi_cols[2]:
        kpi_card("New Badges", "9", delta="• steady", icon="✨")

    badge_html = " ".join(
        [
            pill("Elite", "good"),
            pill("Rising", "neutral"),
            pill("Warning", "warn"),
            pill("At Risk", "bad"),
        ]
    )
    st.markdown(f"<div class='qa-section'>{badge_html}</div>", unsafe_allow_html=True)

    divider()
    st.markdown("### Buttons & form controls")
    button_cols = st.columns(3)
    with button_cols[0]:
        st.button("Primary Action", type="primary")
    with button_cols[1]:
        st.button("Secondary Action")
    with button_cols[2]:
        st.button("Disabled", disabled=True)

    input_cols = st.columns(3)
    with input_cols[0]:
        st.text_input("Text input", placeholder="Placeholder text")
    with input_cols[1]:
        st.selectbox("Selectbox", ["Option A", "Option B", "Option C"])
    with input_cols[2]:
        st.number_input("Number input", value=3, min_value=0, max_value=10)

    st.text_area("Text area", placeholder="Longer description", height=80)

    divider()
    st.markdown("### Tables & deltas")
    table_html = """
    <table class="qa-table">
      <thead>
        <tr>
          <th>Player</th>
          <th>Rating</th>
          <th>Delta</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Jamie</td>
          <td>4.021</td>
          <td><span class="jupr-delta pos">+0.012</span></td>
          <td><span class="jupr-result-badge win">WIN</span></td>
        </tr>
        <tr>
          <td>Morgan</td>
          <td>3.842</td>
          <td><span class="jupr-delta neg">-0.018</span></td>
          <td><span class="jupr-result-badge loss">LOSS</span></td>
        </tr>
        <tr>
          <td>Taylor</td>
          <td>3.910</td>
          <td><span class="jupr-delta zero">+0.000</span></td>
          <td><span class="jupr-result-badge draw">DRAW</span></td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    divider()
    st.markdown("### Links and helper text")
    st.markdown(
        "Need help? Visit the "
        f"<a href='https://example.com' target='_blank'>{html.escape('support page')}</a>."
        "",
        unsafe_allow_html=True,
    )

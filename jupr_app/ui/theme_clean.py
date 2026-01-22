# jupr_app/ui/theme_clean.py
# Clean, sellable UI layer for Streamlit (minimal + durable)
from __future__ import annotations

import html
import streamlit as st


def apply_clean_theme(*, accent_hex: str = "#2F6FED") -> None:
    """
    Call once near the top of streamlit_app.py (after st.set_page_config).
    Keeps it subtle: spacing, typography, cards, pills, buttons, tables.
    """
    css = f"""
    <style>
      :root {{
        --accent: {accent_hex};
        --accent-contrast: #F8FAFC;
        --bg: #F6F7FB;
        --panel: #FDFDFE;
        --text: #111827;
        --muted: #6B7280;
        --border: #E5E7EB;
        --border-strong: #CBD5E1;
        --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        --link: #1D4ED8;
        --link-hover: #1E40AF;
        --focus: rgba(59, 130, 246, 0.45);
        --table-stripe: #F3F4F6;
        --pill-bg: #F8FAFC;
        --radius: 14px;
      }}

      html[data-theme="dark"] {{
        --bg: #0F1218;
        --panel: #161B24;
        --text: #E5E7EB;
        --muted: #9CA3AF;
        --border: #2B3240;
        --border-strong: #3B4557;
        --shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
        --link: #8CB4FF;
        --link-hover: #B3CCFF;
        --focus: rgba(96, 165, 250, 0.55);
        --table-stripe: #1C2230;
        --pill-bg: #1B2230;
      }}

      @media (prefers-color-scheme: dark) {{
        :root {{
          --bg: #0F1218;
          --panel: #161B24;
          --text: #E5E7EB;
          --muted: #9CA3AF;
          --border: #2B3240;
          --border-strong: #3B4557;
          --shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
          --link: #8CB4FF;
          --link-hover: #B3CCFF;
          --focus: rgba(96, 165, 250, 0.55);
          --table-stripe: #1C2230;
          --pill-bg: #1B2230;
        }}
      }}

      /* App background */
      .stApp {{
        background: var(--bg);
        color: var(--text);
      }}

      /* Layout spacing */
      .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2.0rem;
        max-width: 1200px;
      }}

      /* Remove extra padding above first element in many Streamlit versions */
      div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdownContainer"] h1) {{
        margin-top: 0.25rem;
      }}

      /* Headings: calm + consistent */
      h1 {{
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
      }}
      h2, h3 {{
        font-weight: 700;
        letter-spacing: -0.01em;
      }}
      .stCaption, .stMarkdown p {{
        color: var(--muted);
      }}
      a {{
        color: var(--link);
      }}
      a:hover {{
        color: var(--link-hover);
      }}

      /* Panel look for bordered containers (Streamlit uses this wrapper for border=True) */
      [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--panel);
        box-shadow: var(--shadow);
      }}

      /* Buttons: subtle, modern */
      .stButton > button {{
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        font-weight: 650;
        padding: 0.55rem 0.9rem;
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
        transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
      }}
      .stButton > button:hover {{
        border-color: var(--border-strong);
        transform: translateY(-1px);
      }}
      .stButton > button:focus-visible {{
        outline: 3px solid var(--focus);
        outline-offset: 2px;
      }}

      /* Primary buttons: use accent */
      .stButton > button[kind="primary"] {{
        background: var(--accent);
        color: var(--accent-contrast);
        border-color: rgba(0,0,0,0.0);
      }}
      .stButton > button[kind="primary"]:hover {{
        filter: brightness(0.95);
      }}

      /* Inputs: soft borders */
      input, textarea {{
        border-radius: 12px !important;
      }}
      [data-baseweb="select"] > div {{
        border-radius: 12px !important;
      }}
      [data-baseweb="input"] > div {{
        border-radius: 12px !important;
      }}

      /* KPI cards */
      .jupr-kpi {{
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--panel);
        box-shadow: var(--shadow);
        padding: 14px 14px;
        min-height: 86px;
      }}
      .jupr-kpi .label {{
        font-size: 0.86rem;
        color: var(--muted);
        font-weight: 650;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .jupr-kpi .value {{
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text);
        line-height: 1.0;
      }}
      .jupr-kpi .delta {{
        margin-top: 6px;
        font-size: 0.85rem;
        color: var(--muted);
      }}

      /* Pills (status / win% buckets) */
      .jupr-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 750;
        border: 1px solid var(--border);
        background: var(--pill-bg);
        color: var(--text);
        white-space: nowrap;
      }}
      .jupr-pill.good {{ border-color: rgba(16,185,129,0.30); background: rgba(16,185,129,0.10); }}
      .jupr-pill.warn {{ border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.10); }}
      .jupr-pill.bad  {{ border-color: rgba(239,68,68,0.30); background: rgba(239,68,68,0.10); }}
      .jupr-pill.neutral {{ border-color: rgba(59,130,246,0.25); background: rgba(59,130,246,0.08); }}

      /* Dataframe container - keep it calm */
      .stDataFrame {{
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        background: var(--panel);
      }}

      /* Leaderboard table (HTML) */
      .lbtable {{
        width: 100%;
        overflow-x: auto;
      }}
      .lbtable table {{
        width: 100%;
        border-collapse: collapse;
        background: var(--panel);
        color: var(--text);
      }}
      .lbtable th, .lbtable td {{
        padding: 8px;
        border-bottom: 1px solid var(--border);
        text-align: left;
        vertical-align: middle;
        white-space: nowrap;
      }}
      .lbtable th {{
        font-weight: 700;
        background: var(--table-stripe);
      }}
      .lbtable tr:nth-child(even) td {{
        background: var(--table-stripe);
      }}
      .lbtable a {{
        text-decoration: underline;
      }}

      /* Public link button fallback */
      .jupr-link-button {{
        text-decoration: none;
      }}
      .jupr-link-button__btn {{
        padding: 0.5rem 1rem;
        font-size: 1rem;
        border-radius: 0.75rem;
        cursor: pointer;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        font-weight: 650;
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
        transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
      }}
      .jupr-link-button__btn:hover {{
        border-color: var(--border-strong);
        transform: translateY(-1px);
      }}
      .jupr-link-button__btn:focus-visible {{
        outline: 3px solid var(--focus);
        outline-offset: 2px;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def kpi_card(label: str, value: str, *, delta: str = "", icon: str = "") -> None:
    """
    Slot-in KPI card. Keep icons minimal and consistent (use only here + key headers).
    icon can be "", "🏆", "👥" etc. If you want “no emoji,” pass "".
    """
    label_html = html.escape(label)
    value_html = html.escape(value)
    delta_html = html.escape(delta)
    icon_html = html.escape(icon)

    st.markdown(
        f"""
        <div class="jupr-kpi">
          <div class="label">{icon_html}<span>{label_html}</span></div>
          <div class="value">{value_html}</div>
          <div class="delta">{delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "neutral") -> str:
    """
    Returns HTML for a pill to embed in tables (via pandas Styler).
    kind: neutral|good|warn|bad
    """
    t = html.escape(text)
    k = html.escape(kind)
    return f'<span class="jupr-pill {k}">{t}</span>'

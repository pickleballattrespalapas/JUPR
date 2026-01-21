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
        --bg: #F6F7FB;
        --panel: #FFFFFF;
        --text: #111827;
        --muted: #6B7280;
        --border: #E5E7EB;
        --shadow: 0 1px 2px rgba(0,0,0,0.06);
        --radius: 14px;
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
      }}
      .stButton > button:hover {{
        border-color: #D1D5DB;
        transform: translateY(-1px);
      }}

      /* Primary buttons: use accent */
      .stButton > button[kind="primary"] {{
        background: var(--accent);
        color: #FFFFFF;
        border-color: rgba(0,0,0,0.0);
      }}
      .stButton > button[kind="primary"]:hover {{
        filter: brightness(0.97);
      }}

      /* Inputs: soft borders */
      input, textarea {{
        border-radius: 12px !important;
      }}
      [data-baseweb="select"] > div {{
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
        background: #F9FAFB;
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

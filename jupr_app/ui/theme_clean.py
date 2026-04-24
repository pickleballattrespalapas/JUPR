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
        --accent-soft: rgba(47, 111, 237, 0.12);
        --accent-border: rgba(47, 111, 237, 0.30);
        --bg: #F6F7FB;
        --panel: #FDFDFE;
        --text: #111827;
        --muted: #6B7280;
        --text-primary: var(--text);
        --text-secondary: #475569;
        --text-muted: var(--muted);
        --text-inverse: #FFFFFF;
        --text-disabled: #9CA3AF;
        --border: #E5E7EB;
        --border-strong: #CBD5E1;
        --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        --shadow-lg: 0 14px 32px rgba(15, 23, 42, 0.16);
        --link: #1D4ED8;
        --link-hover: #1E40AF;
        --focus: rgba(59, 130, 246, 0.45);
        --table-stripe: #F3F4F6;
        --pill-bg: #F8FAFC;
        --radius: 14px;
        --font-sm: 0.86rem;
        --font-base: 1rem;
        --font-lg: 1.2rem;
        --status-success: #16A34A;
        --status-warning: #D97706;
        --status-danger: #DC2626;
        --status-info: #2563EB;
        --delta-pos: var(--status-success);
        --delta-neg: var(--status-danger);
        --delta-zero: var(--text-muted);
        --result-win-bg: #1F7A6D;
        --result-loss-bg: #5E6F82;
        --result-draw-bg: #B9A874;
        --result-win-text: var(--text-inverse);
        --result-loss-text: var(--text-inverse);
        --result-draw-text: var(--text-primary);
      }}

      html[data-theme="dark"] {{
        --bg: #0F1218;
        --panel: #161B24;
        --text: #E5E7EB;
        --muted: #9CA3AF;
        --text-primary: var(--text);
        --text-secondary: #CBD5E1;
        --text-muted: var(--muted);
        --text-inverse: #0B0F14;
        --text-disabled: #6B7280;
        --border: #2B3240;
        --border-strong: #3B4557;
        --shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
        --shadow-lg: 0 16px 36px rgba(0, 0, 0, 0.55);
        --link: #8CB4FF;
        --link-hover: #B3CCFF;
        --focus: rgba(96, 165, 250, 0.55);
        --table-stripe: #1C2230;
        --pill-bg: #1B2230;
        --accent-soft: rgba(140, 180, 255, 0.16);
        --accent-border: rgba(140, 180, 255, 0.35);
        --status-success: #34D399;
        --status-warning: #FBBF24;
        --status-danger: #F87171;
        --status-info: #93C5FD;
        --delta-pos: var(--status-success);
        --delta-neg: var(--status-danger);
        --delta-zero: var(--text-muted);
        --result-win-bg: #1F9E8F;
        --result-loss-bg: #74839B;
        --result-draw-bg: #CDBE84;
        --result-win-text: var(--text-inverse);
        --result-loss-text: var(--text-inverse);
        --result-draw-text: #1A1D23;
      }}

      @media (prefers-color-scheme: dark) {{
        :root {{
          --bg: #0F1218;
          --panel: #161B24;
          --text: #E5E7EB;
          --muted: #9CA3AF;
          --text-primary: var(--text);
          --text-secondary: #CBD5E1;
          --text-muted: var(--muted);
          --text-inverse: #0B0F14;
          --text-disabled: #6B7280;
          --border: #2B3240;
          --border-strong: #3B4557;
          --shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
          --shadow-lg: 0 16px 36px rgba(0, 0, 0, 0.55);
          --link: #8CB4FF;
          --link-hover: #B3CCFF;
          --focus: rgba(96, 165, 250, 0.55);
          --table-stripe: #1C2230;
          --pill-bg: #1B2230;
          --accent-soft: rgba(140, 180, 255, 0.16);
          --accent-border: rgba(140, 180, 255, 0.35);
          --status-success: #34D399;
          --status-warning: #FBBF24;
          --status-danger: #F87171;
          --status-info: #93C5FD;
          --delta-pos: var(--status-success);
          --delta-neg: var(--status-danger);
          --delta-zero: var(--text-muted);
          --result-win-bg: #1F9E8F;
          --result-loss-bg: #74839B;
          --result-draw-bg: #CDBE84;
          --result-win-text: var(--text-inverse);
          --result-loss-text: var(--text-inverse);
          --result-draw-text: #1A1D23;
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
      .stCaption {{
        color: var(--text-muted);
      }}
      .stMarkdown p,
      .stMarkdown li {{
        color: var(--text-primary);
      }}
      .stMarkdown small {{
        color: var(--text-muted);
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

      /* Topbar */
      .jupr-topbar {{
        position: sticky;
        top: 0.75rem;
        z-index: 10;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 1.1rem;
        border-radius: calc(var(--radius) + 2px);
        border: 1px solid var(--border);
        background: var(--panel);
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
      }}
      .jupr-topbar__title {{
        font-size: var(--font-lg);
        font-weight: 750;
        color: var(--text);
        margin-bottom: 0.1rem;
      }}
      .jupr-topbar__subtitle {{
        font-size: var(--font-sm);
        color: var(--muted);
      }}
      .jupr-topbar__right {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--muted);
        font-size: var(--font-sm);
      }}
      .jupr-topbar-action {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 1.8rem;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        border: 1px solid var(--border-strong);
        background: var(--panel);
        color: var(--text-primary);
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1;
        white-space: nowrap;
        text-decoration: none;
        box-shadow: 0 1px 1px rgba(0, 0, 0, 0.03);
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
      }}
      .jupr-topbar-action:hover {{
        border-color: var(--accent-border);
        background: var(--accent-soft);
        color: var(--text-primary);
        transform: translateY(-1px);
      }}
      .jupr-topbar-action:focus-visible {{
        outline: 3px solid var(--focus);
        outline-offset: 2px;
      }}

      /* Public website shell */
      .jupr-public-shell {{
        width: 100%;
      }}
      .jupr-public-nav {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.9rem;
        margin-bottom: 1.15rem;
        padding: 0.85rem 1rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: color-mix(in srgb, var(--panel) 90%, transparent);
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
      }}
      .jupr-public-brand {{
        display: inline-flex;
        flex-direction: column;
        text-decoration: none;
        color: var(--text-primary);
        font-weight: 800;
        letter-spacing: 0.02em;
      }}
      .jupr-public-brand small {{
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 600;
      }}
      .jupr-public-nav-links {{
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem;
      }}
      .jupr-public-nav-link {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 1px solid var(--border);
        padding: 0.34rem 0.76rem;
        color: var(--text-secondary);
        background: transparent;
        text-decoration: none;
        font-size: 0.84rem;
        font-weight: 700;
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
      }}
      .jupr-public-nav-link:hover {{
        border-color: var(--accent-border);
        background: var(--accent-soft);
        color: var(--text-primary);
        transform: translateY(-1px);
      }}
      .jupr-public-nav-link.active {{
        border-color: var(--accent-border);
        background: var(--accent-soft);
        color: var(--text-primary);
      }}
      .jupr-hero {{
        margin-bottom: 1rem;
        padding: 1.35rem;
      }}
      .jupr-hero-eyebrow {{
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.45rem;
      }}
      .jupr-hero-title {{
        margin: 0 0 0.45rem 0;
        font-size: clamp(1.5rem, 3vw, 2.3rem);
        line-height: 1.15;
      }}
      .jupr-hero-subtitle {{
        margin: 0;
        max-width: 70ch;
        color: var(--text-secondary);
      }}
      .jupr-hero-actions {{
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
      }}
      .jupr-trust-badge {{
        margin-top: 0.85rem;
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        border: 1px solid var(--accent-border);
        background: var(--accent-soft);
        color: var(--text-primary);
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.35rem 0.8rem;
      }}
      .jupr-trust-section {{
        margin-bottom: 0.9rem;
      }}
      .jupr-trust-grid {{
        margin-top: 0.75rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
      }}
      .jupr-trust-card {{
        border: 1px solid var(--border);
        border-radius: calc(var(--radius) - 4px);
        background: color-mix(in srgb, var(--panel) 92%, transparent);
        padding: 0.8rem 0.85rem;
      }}
      .jupr-trust-card h3 {{
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
        color: var(--text-primary);
      }}
      .jupr-trust-card p {{
        margin: 0;
        font-size: 0.88rem;
        color: var(--text-secondary);
      }}
      .jupr-trust-actions {{
        margin-top: 0.8rem;
      }}
      .jupr-home-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
      }}
      .jupr-home-card {{
        height: 100%;
      }}
      .jupr-home-card-title {{
        margin: 0 0 0.35rem 0;
        font-size: 1.02rem;
        color: var(--text-primary);
      }}
      .jupr-home-card-body {{
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.95rem;
      }}
      .jupr-home-card-link {{
        display: block;
        color: inherit;
        text-decoration: none;
      }}
      .jupr-home-card-link:hover {{
        color: inherit;
        text-decoration: none;
      }}
      .jupr-home-card-link:focus-visible {{
        outline: 3px solid var(--focus);
        outline-offset: 3px;
      }}
      .jupr-home-card-cta {{
        margin-top: 0.75rem;
        font-weight: 700;
        color: var(--link);
      }}
      .jupr-home-card-link:hover .jupr-home-card-cta,
      .jupr-home-card-link:focus-visible .jupr-home-card-cta {{
        color: var(--link-hover);
        text-decoration: underline;
        text-underline-offset: 0.15em;
      }}

      /* Cards + sections */
      .jupr-card {{
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--panel);
        box-shadow: var(--shadow);
        padding: 1rem 1.1rem;
      }}
      .jupr-card--hover {{
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
      }}
      .jupr-card--hover:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-border);
      }}
      .jupr-section-title {{
        font-size: var(--font-lg);
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: var(--text);
      }}
      .jupr-divider {{
        height: 1px;
        background: var(--border);
        margin: 0.75rem 0;
      }}

      /* Callouts */
      .jupr-callout {{
        border: 1px solid var(--accent-border);
        background: var(--accent-soft);
        border-radius: var(--radius);
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow);
      }}
      .jupr-callout .title {{
        font-weight: 700;
        font-size: var(--font-base);
        color: var(--text);
        margin-bottom: 0.2rem;
      }}
      .jupr-callout .body {{
        font-size: var(--font-sm);
        color: var(--text);
      }}
      .jupr-callout.info {{
        border-color: var(--accent-border);
        background: var(--accent-soft);
      }}
      .jupr-callout.success {{
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(16, 185, 129, 0.12);
      }}
      .jupr-callout.warn {{
        border-color: rgba(245, 158, 11, 0.35);
        background: rgba(245, 158, 11, 0.12);
      }}
      .jupr-callout.danger {{
        border-color: rgba(239, 68, 68, 0.35);
        background: rgba(239, 68, 68, 0.12);
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
      .stButton > button:disabled {{
        color: var(--text-disabled);
        border-color: var(--border);
        opacity: 0.7;
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
      input::placeholder,
      textarea::placeholder {{
        color: var(--text-muted);
        opacity: 1;
      }}
      [data-baseweb="select"] > div {{
        border-radius: 12px !important;
      }}
      [data-baseweb="input"] > div {{
        border-radius: 12px !important;
      }}
      [data-baseweb="textarea"] > div {{
        border-radius: 12px !important;
      }}
      [data-baseweb="input"] > div,
      [data-baseweb="select"] > div,
      [data-baseweb="textarea"] > div {{
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
      }}
      [data-baseweb="input"] > div:hover,
      [data-baseweb="select"] > div:hover,
      [data-baseweb="textarea"] > div:hover {{
        border-color: var(--border-strong);
      }}
      [data-baseweb="input"] > div:focus-within,
      [data-baseweb="select"] > div:focus-within,
      [data-baseweb="textarea"] > div:focus-within {{
        border-color: var(--accent-border);
        box-shadow: 0 0 0 3px var(--focus);
      }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
      }}
      .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        border: 1px solid var(--border);
        padding: 0.35rem 0.85rem;
        background: var(--panel);
        color: var(--muted);
        font-weight: 650;
        transition: border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
      }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        border-color: var(--accent-border);
        color: var(--text);
        box-shadow: 0 0 0 1px var(--accent-soft);
      }}
      .stTabs [data-baseweb="tab-border"] {{
        display: none;
      }}

      /* Expanders */
      details[data-testid="stExpander"] {{
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: var(--panel);
        box-shadow: var(--shadow);
        padding: 0.35rem 0.6rem;
      }}
      details[data-testid="stExpander"] > summary {{
        font-weight: 650;
        font-size: var(--font-base);
        color: var(--text);
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
        color: var(--text-primary);
        white-space: nowrap;
      }}
      .jupr-pill.good {{ border-color: rgba(16,185,129,0.30); background: rgba(16,185,129,0.10); }}
      .jupr-pill.warn {{ border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.10); }}
      .jupr-pill.bad  {{ border-color: rgba(239,68,68,0.30); background: rgba(239,68,68,0.10); }}
      .jupr-pill.neutral {{ border-color: rgba(59,130,246,0.25); background: rgba(59,130,246,0.08); }}

      /* Result badges + deltas */
      .jupr-result-badge {{
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 650;
        letter-spacing: 0.02em;
      }}
      .jupr-result-badge.win {{
        background: var(--result-win-bg);
        color: var(--result-win-text);
      }}
      .jupr-result-badge.loss {{
        background: var(--result-loss-bg);
        color: var(--result-loss-text);
      }}
      .jupr-result-badge.draw {{
        background: var(--result-draw-bg);
        color: var(--result-draw-text);
      }}
      .jupr-delta.pos {{ color: var(--delta-pos); font-weight: 600; }}
      .jupr-delta.neg {{ color: var(--delta-neg); font-weight: 600; }}
      .jupr-delta.zero {{ color: var(--delta-zero); font-weight: 600; }}

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
      .lbtable tr:hover td {{
        background: var(--accent-soft);
      }}
      .lbtable a {{
        text-decoration: underline;
      }}

      /* Match history table (player page) */
      .match-history-table table {{
        width: 100%;
        border-collapse: collapse;
        background: var(--panel);
        color: var(--text-primary);
      }}
      .match-history-table th,
      .match-history-table td {{
        padding: 8px 10px;
        border-bottom: 1px solid var(--border);
        text-align: left;
        font-size: 0.85rem;
      }}
      .match-history-table th {{
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.72rem;
        color: var(--text-muted);
        background: var(--table-stripe);
      }}
      .match-history-table tr:nth-child(even) td {{
        background: var(--table-stripe);
      }}
      .match-history-table a {{
        color: var(--link);
      }}

      /* Public link button fallback */
      .jupr-link-button {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        min-height: 2.2rem;
        padding: 0.5rem 0.95rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        text-decoration: none;
        font-weight: 650;
        box-shadow: 0 1px 1px rgba(0,0,0,0.04);
        transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
      }}
      .jupr-link-button:hover {{
        border-color: var(--border-strong);
        transform: translateY(-1px);
      }}
      .jupr-link-button:focus-visible {{
        outline: 3px solid var(--focus);
        outline-offset: 2px;
      }}
      .jupr-link-button__btn {{
        all: unset;
        cursor: pointer;
      }}
      .jupr-rules-page {{
        display: grid;
        gap: 0.8rem;
      }}
      .jupr-rules-section {{
        display: grid;
        gap: 0.5rem;
      }}
      .jupr-rules-section h2 {{
        margin: 0;
        font-size: 1.05rem;
        color: var(--text-primary);
      }}
      .jupr-rules-section p {{
        margin: 0;
        color: var(--text-secondary);
      }}
      .jupr-rules-list {{
        margin: 0;
        padding-left: 1.2rem;
        color: var(--text-secondary);
        display: grid;
        gap: 0.35rem;
      }}
      .jupr-rules-callout {{
        border-radius: calc(var(--radius) - 3px);
        border: 1px solid var(--accent-border);
        background: var(--accent-soft);
        padding: 0.7rem 0.8rem;
        color: var(--text-primary);
        font-size: 0.9rem;
        font-weight: 600;
      }}

      @media (max-width: 760px) {{
        .jupr-public-nav {{
          align-items: flex-start;
        }}
        .jupr-public-nav-links {{
          width: 100%;
        }}
        .jupr-home-grid {{
          grid-template-columns: 1fr;
        }}
        .jupr-trust-grid {{
          grid-template-columns: 1fr;
        }}
        .jupr-hero {{
          padding: 1rem;
        }}
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def topbar(title: str, subtitle: str = "", *, right_html: str = "") -> None:
    """
    Render a sticky top bar with a title, subtitle, and optional right-aligned HTML.
    """
    title_html = html.escape(title)
    subtitle_html = html.escape(subtitle)

    st.markdown(
        f"""
        <div class="jupr-topbar">
          <div class="jupr-topbar__left">
            <div class="jupr-topbar__title">{title_html}</div>
            <div class="jupr-topbar__subtitle">{subtitle_html}</div>
          </div>
          <div class="jupr-topbar__right">{right_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(html_body: str, *, hover: bool = False, escape: bool = False) -> None:
    """
    Render a card wrapper for arbitrary HTML content.
    """
    class_name = "jupr-card"
    if hover:
        class_name += " jupr-card--hover"

    body_html = html.escape(html_body) if escape else html_body

    st.markdown(
        f"""
        <div class="{class_name}">
          {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    """
    Render a simple divider line.
    """
    st.markdown('<div class="jupr-divider"></div>', unsafe_allow_html=True)


def callout(kind: str, title: str, body: str) -> None:
    """
    Render a callout block with a title and body.
    """
    variant = kind if kind in {"info", "success", "warn", "danger"} else "info"
    title_html = html.escape(title)
    body_html = html.escape(body)

    st.markdown(
        f"""
        <div class="jupr-callout {variant}">
          <div class="title">{title_html}</div>
          <div class="body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def theme_override_css(mode: str) -> str:
    """
    Return CSS overrides for forcing light/dark tokens in a single session.
    mode: "light" | "dark"
    """
    if mode == "light":
        return """
        :root {
          --bg: #F6F7FB;
          --panel: #FDFDFE;
          --text: #111827;
          --muted: #6B7280;
          --text-primary: var(--text);
          --text-secondary: #475569;
          --text-muted: var(--muted);
          --text-inverse: #FFFFFF;
          --text-disabled: #9CA3AF;
          --border: #E5E7EB;
          --border-strong: #CBD5E1;
          --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
          --shadow-lg: 0 14px 32px rgba(15, 23, 42, 0.16);
          --link: #1D4ED8;
          --link-hover: #1E40AF;
          --focus: rgba(59, 130, 246, 0.45);
          --table-stripe: #F3F4F6;
          --pill-bg: #F8FAFC;
          --accent-soft: rgba(47, 111, 237, 0.12);
          --accent-border: rgba(47, 111, 237, 0.30);
          --status-success: #16A34A;
          --status-warning: #D97706;
          --status-danger: #DC2626;
          --status-info: #2563EB;
          --delta-pos: var(--status-success);
          --delta-neg: var(--status-danger);
          --delta-zero: var(--text-muted);
          --result-win-bg: #1F7A6D;
          --result-loss-bg: #5E6F82;
          --result-draw-bg: #B9A874;
          --result-win-text: var(--text-inverse);
          --result-loss-text: var(--text-inverse);
          --result-draw-text: var(--text-primary);
        }
        """
    if mode == "dark":
        return """
        :root {
          --bg: #0F1218;
          --panel: #161B24;
          --text: #E5E7EB;
          --muted: #9CA3AF;
          --text-primary: var(--text);
          --text-secondary: #CBD5E1;
          --text-muted: var(--muted);
          --text-inverse: #0B0F14;
          --text-disabled: #6B7280;
          --border: #2B3240;
          --border-strong: #3B4557;
          --shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
          --shadow-lg: 0 16px 36px rgba(0, 0, 0, 0.55);
          --link: #8CB4FF;
          --link-hover: #B3CCFF;
          --focus: rgba(96, 165, 250, 0.55);
          --table-stripe: #1C2230;
          --pill-bg: #1B2230;
          --accent-soft: rgba(140, 180, 255, 0.16);
          --accent-border: rgba(140, 180, 255, 0.35);
          --status-success: #34D399;
          --status-warning: #FBBF24;
          --status-danger: #F87171;
          --status-info: #93C5FD;
          --delta-pos: var(--status-success);
          --delta-neg: var(--status-danger);
          --delta-zero: var(--text-muted);
          --result-win-bg: #1F9E8F;
          --result-loss-bg: #74839B;
          --result-draw-bg: #CDBE84;
          --result-win-text: var(--text-inverse);
          --result-loss-text: var(--text-inverse);
          --result-draw-text: #1A1D23;
        }
        """
    return ""


# Demo snippet (comment-only):
#
# apply_clean_theme()
# topbar("JUPR Dashboard", "Season overview", right_html="<a href='#'>Settings</a>")
# kpi_card("Total Matches", "128", delta="+12 vs last month")
# callout("info", "Heads up", "Rankings refresh nightly at 2am.")
# divider()
# card("<strong>Card content</strong>", hover=True)

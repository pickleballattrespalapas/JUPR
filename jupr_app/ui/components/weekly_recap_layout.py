from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st


def render_weekly_recap(recap: dict, *, print_view: bool, title_override: str | None = None) -> None:
    week_start = recap.get("week_start")
    week_end = recap.get("week_end")
    title = title_override or "Tres Palapas Weekly Recap"
    theme = (recap.get("meta") or {}).get("print_theme", "classic")

    date_label = ""
    if week_start and week_end:
        try:
            start_dt = datetime.fromisoformat(str(week_start))
            end_dt = datetime.fromisoformat(str(week_end))
            date_label = f"{start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d')}"
        except Exception:
            date_label = f"{week_start} – {week_end}"

    numbers = recap.get("numbers", {})
    spotlight = recap.get("spotlight", [])
    around = recap.get("around_club", {})
    looking_ahead = recap.get("looking_ahead", []) or []

    league_items = around.get("leagues", []) or []
    rr_items = around.get("round_robins", []) or []

    spotlight_items = []
    for item in (spotlight or []):
        if not isinstance(item, dict):
            continue
        label = (item or {}).get("label", "")
        display = (item or {}).get("display", "")
        description = (item or {}).get("description", "") or ""
        desc_html = f"<span class=\"award-desc\">{escape(description)}</span><br/>" if description else ""
        spotlight_items.append(
            "<li>"
            f"<strong>{label}</strong>: {desc_html}"
            f"{display}"
            "</li>"
        )
    spotlight_html = "".join(spotlight_items)
    leagues_html = "".join(
        _render_event_block((item or {}).get("league_name", "League"), (item or {}).get("highlights", []))
        for item in (league_items or [])
        if isinstance(item, dict)
    )
    rr_html = "".join(
        _render_event_block((item or {}).get("event_name", "Pop-Up Event"), (item or {}).get("highlights", []))
        for item in (rr_items or [])
        if isinstance(item, dict)
    )
    looking_ahead_html = "".join(
        f"<li>{item}</li>" for item in (looking_ahead or [])
        if str(item).strip() != ""
    )

    print_mode_notice = "<!-- PRINT MODE -->" if print_view else ""
    print_view_css = ".no-print { display: none !important; }" if print_view else ""

    html = f"""
    <style>
      {print_view_css}
      {_css_common_print()}
      {_css_tokens_baja_v2()}
      {_css_tokens_newsletter_sep()}
      {_css_layout_overrides_for_theme(theme)}
      .weekly-recap {{
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        max-width: 900px;
        margin: 0 auto;
        padding: 16px 20px 24px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--card);
        --ink: #111827;
        --muted: #475569;
        --border: #e5e7eb;
        --bg: #ffffff;
        --card: #ffffff;
        --accent: #111827;
        --accent2: #6b7280;
        --stat-bg: #f3f4f6;
        --pill-bg: #f3f4f6;
      }}
      .weekly-hero {{
        margin-bottom: 12px;
      }}
      .weekly-hero__bar {{
        display: none;
        height: 6px;
        border-radius: 999px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
      }}
      .weekly-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 3px solid var(--accent);
        padding-bottom: 10px;
        gap: 12px;
      }}
      .weekly-title {{
        font-size: 28px;
        font-weight: 700;
      }}
      .weekly-subtitle {{
        font-size: 12.5px;
        color: var(--muted);
        margin-top: 4px;
      }}
      .weekly-range-pill {{
        font-size: 12px;
        font-weight: 600;
        padding: 6px 12px;
        border-radius: 999px;
        background: var(--pill-bg);
        color: var(--ink);
        border: 1px solid var(--border);
        white-space: nowrap;
      }}
      .numbers-strip {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 16px;
      }}
      .number-card {{
        background: var(--stat-bg);
        padding: 10px 8px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--border);
        border-top: 4px solid var(--accent);
        box-shadow: none;
      }}
      .number-value {{
        font-size: 20px;
        font-weight: 700;
      }}
      .number-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}
      .section {{
        margin-top: 14px;
      }}
      .section h3 {{
        font-size: 18px;
        margin: 0 0 8px;
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      .section h3::before {{
        content: "";
        width: 10px;
        height: 10px;
        background: var(--accent2);
        border-radius: 3px;
        display: inline-block;
      }}
      .subsection h4 {{
        font-size: 13px;
        margin: 10px 0 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--muted);
      }}
      .event-block {{
        margin-bottom: 8px;
      }}
      .event-name {{
        font-weight: 600;
        font-size: 13px;
      }}
      ul.compact {{
        margin: 6px 0 8px 18px;
        padding: 0;
      }}
      ul.compact li {{
        margin-bottom: 6px;
        font-size: 13px;
      }}
      .award-desc {{
        color: var(--muted);
        font-size: 12.5px;
      }}
      .looking-ahead li {{
        font-size: 13px;
      }}
      a {{
        color: var(--accent);
        text-decoration: underline;
      }}
    </style>
    {print_mode_notice}
    <div class=\"weekly-recap theme-{theme}\">
      <div class=\"weekly-hero\">
        <div class=\"weekly-hero__bar\"></div>
        <div class=\"weekly-header\">
          <div>
            <div class=\"weekly-title\">{title}</div>
            <div class=\"weekly-subtitle\">Tres Palapas Baja Pickleball Resort • Los Barriles, BCS</div>
          </div>
          <div class=\"weekly-range-pill\">{date_label}</div>
        </div>
      </div>
      <div class=\"numbers-strip\">
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('matches', 0)}</div><div class=\"number-label\">Matches</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('players', 0)}</div><div class=\"number-label\">Players</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('leagues', 0)}</div><div class=\"number-label\">Leagues</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('round_robins', 0)}</div><div class=\"number-label\">Pop-Ups</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('new_faces', 0)}</div><div class=\"number-label\">New Faces</div></div>
      </div>
      <div class=\"section\">
        <h3>Spotlight Reel</h3>
        <ul class=\"compact\">
          {spotlight_html}
        </ul>
      </div>
      <div class=\"section\">
        <h3>Around the Club</h3>
        <div class=\"subsection\">
          <h4>Leagues</h4>
          {leagues_html}
        </div>
        <div class=\"subsection\">
          <h4>Round Robins</h4>
          {rr_html}
        </div>
      </div>
      <div class=\"section\">
        <h3>Looking Ahead</h3>
        <ul class=\"compact looking-ahead\">
          {looking_ahead_html}
        </ul>
      </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def _render_event_block(name: str, highlights: list[dict]) -> str:
    safe = [h for h in (highlights or []) if isinstance(h, dict)]
    items = "".join(f"<li>{h.get('display','')}</li>" for h in safe if str(h.get("display","")).strip() != "")
    return f"<div class='event-block'><div class='event-name'>{name}</div><ul class='compact'>{items}</ul></div>"


def _css_tokens_baja_v2() -> str:
    return """
      .weekly-recap.theme-baja_v2 {
        --ink: #111827;
        --muted: #475569;
        --border: #e5e7eb;
        --bg: #ffffff;
        --card: #ffffff;
        --sand: #f6e7c6;
        --ocean: #0ea5a4;
        --sunset: #ff6a3d;
        --accent: var(--ocean);
        --accent2: var(--sunset);
        --stat-bg: var(--sand);
        --pill-bg: #ffffff;
      }
      .weekly-recap.theme-baja_v2 .weekly-hero__bar {
        display: block;
        background: linear-gradient(90deg, #0ea5a4, #ff6a3d);
      }
      .weekly-recap.theme-baja_v2 .weekly-range-pill {
        background: #fff7ed;
      }
    """


def _css_tokens_newsletter_sep() -> str:
    return """
      .weekly-recap.theme-newsletter_sep {
        --ink: #111827;
        --muted: #475569;
        --border: #e5e7eb;
        --bg: #ffffff;
        --card: #ffffff;
        --accent: #1d4ed8;
        --accent2: #0f766e;
        --soft: rgba(29, 78, 216, 0.08);
        --stat-bg: var(--soft);
        --pill-bg: var(--soft);
      }
    """


def _css_common_print() -> str:
    return """
      * {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }
      @media print {
        body {
          margin: 0;
        }
        .weekly-recap {
          border: none;
          margin: 0;
          padding: 0.4in 0.5in;
          box-shadow: none;
        }
        .weekly-hero__bar {
          box-shadow: none;
        }
        .number-card {
          box-shadow: none;
        }
        .no-print {
          display: none !important;
        }
      }
    """


def _css_layout_overrides_for_theme(theme: str) -> str:
    if theme == "newsletter_sep":
        return """
      .weekly-recap.theme-newsletter_sep {
        background: var(--card);
      }
      """
    if theme == "baja_v2":
        return """
      .weekly-recap.theme-baja_v2 {
        background: var(--card);
      }
      """
    return ""

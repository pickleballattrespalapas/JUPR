from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st
import streamlit.components.v1 as components

from jupr_app.domain.recaps.weekly_recap import (
    DEFAULT_AROUND_LEAGUE_DESCRIPTION,
    DEFAULT_AROUND_RR_DESCRIPTION,
    build_around_descriptions,
)

def build_weekly_recap_html(
    recap: dict,
    *,
    print_view: bool,
    title_override: str | None = None,
) -> str:
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
    around_descriptions = recap.get("around_descriptions") or build_around_descriptions(around)
    looking_ahead = recap.get("looking_ahead", []) or []

    league_items = around.get("leagues", []) or []
    rr_items = around.get("round_robins", []) or []

    spotlight_cards_html = "".join(
        _render_award_card(item, theme=theme)
        for item in (spotlight or [])
        if isinstance(item, dict)
    )
    around_cards_html = "".join(
        _render_event_card(
            title=(item or {}).get("league_name", "League"),
            description=_resolve_around_description(item or {}, around_descriptions, kind="league"),
            highlights=(item or {}).get("highlights", []),
            kind="league",
        )
        for item in (league_items or [])
        if isinstance(item, dict)
    )
    around_cards_html += "".join(
        _render_event_card(
            title=(item or {}).get("event_name", "Pop-Up Event"),
            description=_resolve_around_description(item or {}, around_descriptions, kind="rr"),
            highlights=(item or {}).get("highlights", []),
            kind="rr",
        )
        for item in (rr_items or [])
        if isinstance(item, dict)
    )
    looking_ahead_html = "".join(
        f"<li>{escape(str(item))}</li>" for item in (looking_ahead or [])
        if str(item).strip() != ""
    )
    looking_ahead_accent = "accent-sunset"
    if theme == "baja_v2":
        looking_ahead_accent = "accent-sand"
    elif theme == "newsletter_sep":
        looking_ahead_accent = "accent-ocean"
    # Only render "Looking Ahead" when there is at least one non-empty bullet.
    looking_card_html = ""
    if looking_ahead_html.strip():
        looking_card_html = _render_simple_card(
            title="Looking Ahead",
            body_html=f"<ul class='compact looking-ahead'>{looking_ahead_html}</ul>",
            accent_class=looking_ahead_accent,
        )
    looking_section_html = f"<div class='section'>{looking_card_html}</div>" if looking_card_html else ""

    print_mode_notice = "<!-- PRINT MODE -->" if print_view else ""
    print_view_css = ".no-print { display: none !important; }" if print_view else ""

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{escape(title)}</title>
        <style>
          {print_view_css}
          {_css_pdf_safe()}
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
          .award-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
          }}
          .event-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
          }}
          .award-card {{
            border: 1px solid var(--border, #e5e7eb);
            border-left: 6px solid var(--accent, #111827);
            border-radius: 12px;
            background: var(--card, #ffffff);
            padding: 12px;
          }}
          .event-card {{
            border: 1px solid var(--border, #e5e7eb);
            border-left: 6px solid var(--accent, #111827);
            border-radius: 12px;
            background: var(--card, #ffffff);
            padding: 12px;
          }}
          .section-card.accent-sunset {{
            border-left-color: var(--sunset, var(--accent2, var(--accent)));
          }}
          .section-card.accent-ocean {{
            border-left-color: var(--ocean, var(--accent));
          }}
          .section-card.accent-sand {{
            border-left-color: var(--sand-accent, var(--accent));
          }}
          .section-card {{
            border: 1px solid var(--border, #e5e7eb);
            border-left: 5px solid var(--accent, #111827);
            border-radius: 12px;
            background: var(--card, #ffffff);
            padding: 12px;
          }}
          .event-card__title,
          .section-card__title {{
            font-weight: 700;
            font-size: 14px;
          }}
          .event-card__desc {{
            margin-top: 6px;
            color: var(--muted, #475569);
            font-size: 12.5px;
            line-height: 1.35;
          }}
          .event-card__body,
          .section-card__body {{
            margin-top: 8px;
            font-size: 13px;
            line-height: 1.45;
          }}
          .award-card__title {{
            font-weight: 800;
            font-size: 14px;
          }}
          .award-card__desc {{
            margin-top: 6px;
            color: var(--muted, #475569);
            font-size: 12.5px;
            line-height: 1.35;
          }}
          .award-card__body {{
            margin-top: 10px;
            font-size: 13px;
            line-height: 1.45;
          }}
          .award-card.accent-ocean {{
            border-left-color: var(--ocean, var(--accent));
          }}
          .award-card.accent-sunset {{
            border-left-color: var(--sunset, var(--accent2, var(--accent)));
          }}
          .award-card.accent-ink {{
            border-left-color: var(--ink, #111827);
          }}
          .award-card.accent-sand {{
            border-left-color: var(--sand-accent, var(--accent));
          }}
          .event-card.accent-ocean {{
            border-left-color: var(--ocean, var(--accent));
          }}
          .event-card.accent-sunset {{
            border-left-color: var(--sunset, var(--accent2, var(--accent)));
          }}
          .event-card.accent-ink {{
            border-left-color: var(--ink, #111827);
          }}
          @media (min-width: 820px) {{
            .award-grid {{
              grid-template-columns: 1fr 1fr;
            }}
            .event-grid {{
              grid-template-columns: 1fr 1fr;
            }}
          }}
          @media print {{
            .award-grid {{
              grid-template-columns: 1fr 1fr;
            }}
            .event-grid {{
              grid-template-columns: 1fr 1fr;
            }}
            /* ---- Pagination safety: prevent cards/sections from splitting across pages ---- */
            .section,
            .award-card,
            .event-card,
            .section-card,
            .number-card {{
              break-inside: avoid;
              page-break-inside: avoid;
            }}

            /* ---- Print tightening: Spotlight cards become more bulletin-friendly ---- */
            .weekly-recap {{
              padding: 0.35in 0.45in;
            }}
            .section {{
              margin-top: 10px;
            }}
            .award-grid,
            .event-grid {{
              gap: 8px;
            }}
            .award-card,
            .event-card,
            .section-card {{
              padding: 10px;
            }}
            .award-card__desc {{
              font-size: 11.5px;
              line-height: 1.25;
              /* Clamp long descriptions so Spotlight doesn't bloat the print/PDF */
              display: -webkit-box;
              -webkit-line-clamp: 2;
              -webkit-box-orient: vertical;
              overflow: hidden;
            }}
            .award-card__body {{
              margin-top: 6px;
              font-size: 12.5px;
              line-height: 1.35;
            }}
            ul.compact li {{
              margin-bottom: 4px;
              font-size: 12.5px;
            }}
          }}
          .looking-ahead li {{
            font-size: 13px;
          }}
          a {{
            color: var(--accent);
            text-decoration: underline;
          }}
        </style>
      </head>
      <body>
        {print_mode_notice}
        <div class="weekly-recap theme-{theme}">
          <div class="weekly-hero">
            <div class="weekly-hero__bar"></div>
            <div class="weekly-header">
              <div>
                <div class="weekly-title">{title}</div>
                <div class="weekly-subtitle">Tres Palapas Baja Pickleball Resort • Los Barriles, BCS</div>
              </div>
              <div class="weekly-range-pill">{date_label}</div>
            </div>
          </div>
          <div class="numbers-strip">
            <div class="number-card"><div class="number-value">{numbers.get('matches', 0)}</div><div class="number-label">Matches</div></div>
            <div class="number-card"><div class="number-value">{numbers.get('players', 0)}</div><div class="number-label">Players</div></div>
            <div class="number-card"><div class="number-value">{numbers.get('leagues', 0)}</div><div class="number-label">Leagues</div></div>
            <div class="number-card"><div class="number-value">{numbers.get('round_robins', 0)}</div><div class="number-label">Pop-Ups</div></div>
            <div class="number-card"><div class="number-value">{numbers.get('new_faces', 0)}</div><div class="number-label">New Faces</div></div>
          </div>
          <div class="section">
            <h3>Spotlight Reel</h3>
            <div class="award-grid">
              {spotlight_cards_html}
            </div>
          </div>
          <div class="section">
            <h3>Around the Club</h3>
            <div class="event-grid">
              {around_cards_html}
            </div>
          </div>
          {looking_section_html}
        </div>
      </body>
    </html>
    """
    return html


def render_weekly_recap(recap: dict, *, print_view: bool, title_override: str | None = None) -> None:
    html = build_weekly_recap_html(recap, print_view=print_view, title_override=title_override)
    components.html(html, height=None, scrolling=False)


def _render_award_card(item: dict, *, theme: str) -> str:
    label = escape(item.get("label", ""))
    desc = escape(item.get("description", "") or "")
    body = item.get("display", "")
    key = item.get("key", "")
    accent_class = {
        "TOP_PERFORMER_WEEK": "accent-ocean",
        "BIGGEST_JUMP_WEEK": "accent-sunset",
        "GIANT_SLAYER_WEEK": "accent-ink",
        "GRIND_WEEK": "accent-sand",
        "PERFECT_RUN": "accent-ocean",
    }.get(key, "accent-ink")
    desc_html = f"<div class='award-card__desc'>{desc}</div>" if desc else ""
    return (
        f"<div class='award-card {accent_class}'>"
        "<div class='award-card__head'>"
        f"<div class='award-card__title'>{label}</div>"
        f"{desc_html}"
        "</div>"
        f"<div class='award-card__body'>{body}</div>"
        "</div>"
    )


def _resolve_around_description(item: dict, around_descriptions: dict[str, str], *, kind: str) -> str:
    description = str(item.get("description", "") or "").strip()
    if description:
        return description
    if kind == "league":
        name = str(item.get("league_name", "") or "").strip()
        desc_key = item.get("desc_key") or f"LEAGUE:{name}"
        return around_descriptions.get(desc_key, DEFAULT_AROUND_LEAGUE_DESCRIPTION)
    event_id = str(item.get("event_id", "") or "").strip()
    desc_key = item.get("desc_key") or f"RR:{event_id}"
    return around_descriptions.get(desc_key, DEFAULT_AROUND_RR_DESCRIPTION)


def _render_event_card(title: str, description: str, highlights: list[dict], *, kind: str) -> str:
    safe = [h for h in (highlights or []) if isinstance(h, dict)]
    items = "".join(
        f"<li>{escape(str(h.get('display', '')))}</li>"
        for h in safe
        if str(h.get("display", "")).strip() != ""
    )
    if kind == "league":
        accent_class = "accent-ocean"
    elif kind == "rr":
        accent_class = "accent-sunset"
    else:
        accent_class = "accent-ink"
    desc_html = f"<div class='event-card__desc'>{escape(description)}</div>" if description else ""
    return (
        f"<div class='event-card {accent_class}'>"
        f"<div class='event-card__title'>{escape(title)}</div>"
        f"{desc_html}"
        f"<div class='event-card__body'><ul class='compact'>{items}</ul></div>"
        "</div>"
    )


def _render_simple_card(title: str, body_html: str, *, accent_class: str) -> str:
    return (
        f"<div class='section-card {accent_class}'>"
        f"<div class='section-card__title'>{escape(title)}</div>"
        f"<div class='section-card__body'>{body_html}</div>"
        "</div>"
    )


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
        --sand-accent: #d7a648;
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
      .weekly-recap.theme-baja_v2 .award-card.accent-sand {
        background: rgba(246, 231, 198, 0.55);
      }
      .weekly-recap.theme-baja_v2 .award-card.accent-ocean {
        background: rgba(14, 165, 164, 0.06);
      }
      .weekly-recap.theme-baja_v2 .award-card.accent-sunset {
        background: rgba(255, 106, 61, 0.06);
      }
      .weekly-recap.theme-baja_v2 .event-card.accent-ocean {
        background: rgba(14, 165, 164, 0.05);
      }
      .weekly-recap.theme-baja_v2 .event-card.accent-sunset {
        background: rgba(255, 106, 61, 0.05);
      }
      .weekly-recap.theme-baja_v2 .section-card {
        background: rgba(255, 255, 255, 0.85);
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
      .weekly-recap.theme-newsletter_sep .award-card.accent-ocean,
      .weekly-recap.theme-newsletter_sep .award-card.accent-sunset {
        background: var(--soft, rgba(29, 78, 216, 0.06));
      }
      .weekly-recap.theme-newsletter_sep .event-card.accent-ocean,
      .weekly-recap.theme-newsletter_sep .event-card.accent-sunset {
        background: var(--soft, rgba(29, 78, 216, 0.06));
      }
      .weekly-recap.theme-newsletter_sep .section-card {
        background: var(--soft, rgba(29, 78, 216, 0.06));
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


def _css_pdf_safe() -> str:
    return """
      @page {
        size: Letter;
        margin: 0.5in;
      }
      html,
      body {
        background: #fff;
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

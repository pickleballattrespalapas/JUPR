from __future__ import annotations

import html

import pandas as pd
import streamlit as st


def _inject_styles() -> None:
    st.markdown(
        """
<style>
  .player-digest-wrap { max-width: 980px; margin: 0 auto; }
  .player-digest { font-family: 'Inter', sans-serif; border: 1px solid gainsboro; border-radius: 14px; padding: 18px; }
  .digest-card { border-radius: 16px; padding: 16px 18px; margin-bottom: 14px; background: var(--secondary-background-color); }
  .digest-header { background: linear-gradient(135deg, #FFF3E6, #FFE7CF); border-left: 6px solid #FF7A00; color: #111; }
  .digest-title { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
  .digest-subtitle { opacity: 0.8; font-size: 14px; }
  .digest-metrics { display: grid; grid-template-columns: repeat(auto-fit,minmax(120px,1fr)); gap: 8px; }
  .digest-metric { border-radius: 12px; background: var(--background-color); padding: 10px; text-align: center; }
  .digest-metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
  .digest-metric-value { font-size: 20px; font-weight: 700; }
  .digest-section-title { font-size: 17px; font-weight: 700; margin-bottom: 8px; }
  .digest-soft { background: linear-gradient(135deg, #EAF4FF, #D8E9FF); border-left: 5px solid #1976D2; color: #111; }
  .digest-soft-green { background: linear-gradient(135deg, #E8FFF4, #D2F7E7); border-left: 5px solid #1B9E77; color: #111; }
  .digest-list { margin: 0; padding-left: 18px; }
  .digest-list li { margin: 6px 0; }
  .digest-row { border-bottom: 1px solid rgba(128,128,128,.25); padding: 7px 0; }
  .digest-row:last-child { border-bottom: none; }
  .digest-muted { opacity: .8; font-size: 13px; }
  .digest-cta { text-align:center; }
</style>
""",
        unsafe_allow_html=True,
    )


def _esc(value) -> str:
    return html.escape(str(value or ""))


def _render_people_rows(items: list[dict], empty_text: str) -> str:
    if not items:
        return f"<div class='digest-muted'>{_esc(empty_text)}</div>"
    rows = []
    for item in items[:3]:
        rows.append(
            "<div class='digest-row'>"
            f"<strong>{_esc(item.get('player_name') or ('Player #' + str(item.get('player_id') or '')))}</strong>"
            f"<div class='digest-muted'>{int(item.get('matches') or 0)} matches · {_esc(item.get('record') or '0-0')} · {int(item.get('point_diff') or 0):+d} point diff</div>"
            "</div>"
        )
    return "".join(rows)


def render_player_digest(digest: dict) -> None:
    digest = digest or {}
    summary = digest.get("summary") or {}
    numbers = digest.get("numbers_cards") or []
    highlights = digest.get("highlights") or []
    glance = digest.get("week_at_a_glance") or []
    leagues = digest.get("league_breakdown") or []
    people = digest.get("people") or {}
    badges = digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    notable = digest.get("notable_results") or []

    _inject_styles()

    player_name = _esc(digest.get("player_name") or "Verified Player")
    display_range = _esc(digest.get("display_range") or "")
    overall_after = summary.get("overall_jupr_after")
    overall_delta = summary.get("overall_delta")

    try:
        overall_text = f"{float(overall_after):.3f}"
    except Exception:
        overall_text = "—"
    try:
        delta_text = f"{float(overall_delta):+0.4f}"
    except Exception:
        delta_text = "—"

    metric_html = "".join(
        f"<div class='digest-metric'><div class='digest-metric-value'>{_esc(card.get('value'))}</div><div class='digest-metric-label'>{_esc(card.get('label'))}</div></div>"
        for card in numbers
    )

    st.markdown(
        f"""
<div class="player-digest-wrap">
  <div class="player-digest">
    <div class="digest-card digest-header">
      <div class="digest-title">{player_name}</div>
      <div class="digest-subtitle">Verified player update · {display_range}</div>
      <div style="margin-top:8px;"><strong>Current JUPR:</strong> {overall_text} &nbsp;·&nbsp; <strong>Range Δ:</strong> {delta_text}</div>
    </div>
    <div class="digest-card">
      <div class="digest-section-title">Quick Numbers</div>
      <div class="digest-metrics">{metric_html}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("#### Week at a glance")
        if glance:
            for line in glance:
                st.write(f"• {line}")
        else:
            st.caption("Quiet week — no match activity in this date range.")

    chart_points = (digest.get("chart") or {}).get("points") or []
    with st.container(border=True):
        st.markdown("#### Overall JUPR Trend")
        if chart_points:
            chart_df = pd.DataFrame(chart_points)
            chart_df["date"] = pd.to_datetime(chart_df.get("date"), errors="coerce")
            chart_df = chart_df.dropna(subset=["date", "overall_after"]).sort_values("date")
            if not chart_df.empty:
                st.line_chart(chart_df.set_index("date")["overall_after"], use_container_width=True)
                st.caption(f"{len(chart_df)} chart points in selected range")
            else:
                st.caption("No chartable points in selected range.")
        else:
            st.caption("No chart points available in selected date range.")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("#### Highlights")
            for line in highlights[:6] or ["No highlights available for this period."]:
                st.write(f"• {line}")
        with st.container(border=True):
            st.markdown("#### League Breakdown")
            if leagues:
                for row in leagues[:8]:
                    st.write(
                        f"**{row.get('league_name')}** — {int(row.get('matches') or 0)} matches · {row.get('record') or '0-0'} · Δ {float(row.get('overall_delta') or 0):+0.4f}"
                    )
            else:
                st.caption("No league data in this date range.")

    with right:
        with st.container(border=True):
            st.markdown("#### Played With")
            st.markdown(
                _render_people_rows(people.get("top_partners") or [], "No partner data in this date range."),
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.markdown("#### Faced Most")
            st.markdown(
                _render_people_rows(people.get("top_opponents") or [], "No opponent data in this date range."),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown("#### Awards")
        badge_names = [str(item.get("name") or item.get("badge_id") or "Badge") for item in badges[:6]]
        trophy_names = [str(item.get("tournament_name") or item.get("league_name") or "Trophy") for item in trophies[:6]]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Badges earned**")
            for name in badge_names or ["None this period."]:
                st.write(f"• {name}")
        with c2:
            st.markdown("**Trophies earned**")
            for name in trophy_names or ["None this period."]:
                st.write(f"• {name}")

    if notable:
        with st.container(border=True):
            st.markdown("#### Notable Results")
            for item in notable[:5]:
                st.write(f"**{item.get('title')}:** {item.get('detail')}")

    links = digest.get("links") or {}
    profile_link = links.get("player_profile")
    if profile_link:
        st.markdown(f"[View player profile]({_esc(profile_link)})")

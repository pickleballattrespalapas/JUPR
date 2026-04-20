from __future__ import annotations

import html


def _s(value) -> str:
    return str(value or "").strip()


def build_player_update_email_subject(digest_json: dict) -> str:
    player_name = _s((digest_json or {}).get("player_name")) or "Verified Player"
    display_range = _s((digest_json or {}).get("display_range"))
    if display_range:
        return f"{player_name} verified update · {display_range}"
    return f"{player_name} verified update"


def build_player_update_email_html(digest_json: dict, chart_cid: str | None) -> str:
    digest = digest_json or {}
    summary = digest.get("summary") or {}
    badges = digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    highlights = digest.get("highlights") or []
    links = digest.get("links") or {}

    player_name = html.escape(_s(digest.get("player_name")) or "Verified Player")
    display_range = html.escape(_s(digest.get("display_range")))
    overall = summary.get("overall_jupr_after")
    try:
        overall_text = f"{float(overall):.3f}"
    except Exception:
        overall_text = "N/A"

    record = html.escape(_s(summary.get("record")) or "0-0")
    matches_played = int(summary.get("matches_played") or 0)

    badge_items = "".join(
        f"<li>{html.escape(_s(item.get('name')) or _s(item.get('badge_id')) or 'Badge')}</li>"
        for item in badges[:10]
    )
    trophy_items = "".join(
        f"<li>{html.escape(_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy')}</li>"
        for item in trophies[:10]
    )
    highlight_items = "".join(f"<li>{html.escape(_s(line))}</li>" for line in highlights[:10])

    profile_link = html.escape(_s(links.get("player_profile")))
    unsubscribe_link = html.escape(_s(links.get("unsubscribe")))

    chart_block = ""
    if chart_cid:
        chart_block = (
            "<h3>Overall JUPR trend</h3>"
            f"<img alt=\"Overall JUPR trend\" src=\"cid:{html.escape(chart_cid)}\" "
            "style=\"max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;\"/>"
        )
    else:
        chart_block = "<p><em>Chart unavailable for this date range (fewer than 2 points).</em></p>"

    return f"""
    <html>
      <body style=\"font-family:Arial,sans-serif;line-height:1.5;color:#111;\">
        <h2>Verified Player Update</h2>
        <p><strong>Player:</strong> {player_name}<br/>
           <strong>Date range:</strong> {display_range}<br/>
           <strong>Current overall JUPR:</strong> {overall_text}</p>

        <p><strong>Weekly record:</strong> {record}<br/>
           <strong>Matches played:</strong> {matches_played}</p>

        {chart_block}

        <h3>Badges earned</h3>
        {('<ul>' + badge_items + '</ul>') if badge_items else '<p>None this period.</p>'}

        <h3>Trophies earned</h3>
        {('<ul>' + trophy_items + '</ul>') if trophy_items else '<p>None this period.</p>'}

        <h3>Highlights</h3>
        {('<ul>' + highlight_items + '</ul>') if highlight_items else '<p>No highlights available.</p>'}

        <p><a href=\"{profile_link}\">View player profile</a></p>
        <p style=\"font-size:12px;color:#444;\">To stop these updates, <a href=\"{unsubscribe_link}\">unsubscribe in one click</a>.</p>
      </body>
    </html>
    """.strip()


def build_player_update_email_text(digest_json: dict) -> str:
    digest = digest_json or {}
    summary = digest.get("summary") or {}
    badges = digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    highlights = digest.get("highlights") or []
    links = digest.get("links") or {}

    badge_lines = [f"- {_s(item.get('name')) or _s(item.get('badge_id')) or 'Badge'}" for item in badges[:10]]
    trophy_lines = [
        f"- {_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy'}"
        for item in trophies[:10]
    ]
    highlight_lines = [f"- {_s(line)}" for line in highlights[:10]]

    return "\n".join(
        [
            "Verified Player Update",
            f"Player: {_s(digest.get('player_name'))}",
            f"Date range: {_s(digest.get('display_range'))}",
            f"Current overall JUPR: {summary.get('overall_jupr_after')}",
            f"Weekly record: {_s(summary.get('record')) or '0-0'}",
            f"Matches played: {int(summary.get('matches_played') or 0)}",
            "",
            "Badges earned:",
            *(badge_lines or ["- None this period."]),
            "",
            "Trophies earned:",
            *(trophy_lines or ["- None this period."]),
            "",
            "Highlights:",
            *(highlight_lines or ["- No highlights available."]),
            "",
            f"Profile link: {_s(links.get('player_profile'))}",
            f"Unsubscribe: {_s(links.get('unsubscribe'))}",
        ]
    ).strip()

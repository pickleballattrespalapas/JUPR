from __future__ import annotations

import html


def _s(value) -> str:
    return str(value or "").strip()


def _num(value, digits: int = 3, prefix_sign: bool = False, default: str = "N/A") -> str:
    try:
        if prefix_sign:
            return f"{float(value):+.{digits}f}"
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def _people_html(items: list[dict], empty_text: str) -> str:
    if not items:
        return f"<tr><td style='padding:4px 0;color:#5A6678;font-size:13px;'>{html.escape(empty_text)}</td></tr>"
    rows = []
    for item in items[:3]:
        name = _s(item.get("player_name")) or f"Player #{_s(item.get('player_id'))}"
        matches = int(item.get("matches") or 0)
        record = _s(item.get("record")) or "0-0"
        rows.append(
            f"<tr><td style='padding:4px 0;font-size:13px;line-height:1.35;'><strong>{html.escape(name)}</strong>"
            f"<span style='color:#5A6678;'> · {matches} matches · {html.escape(record)}</span></td></tr>"
        )
    return "".join(rows)


def _badge_line(item: dict) -> str:
    name = _s(item.get("name")) or _s(item.get("badge_id")) or "Badge"
    count = int(item.get("count") or 1)
    label = f"{name} x{count}" if count > 1 else name
    desc = _s(item.get("description")) or "Award earned during this update window."
    return (
        "<tr><td style='padding:4px 0;'>"
        f"<div><strong>{html.escape(label)}</strong></div>"
        f"<div style='color:#5A6678;font-size:12px;line-height:1.3;'>{html.escape(desc)}</div>"
        "</td></tr>"
    )


def build_player_update_email_subject(digest_json: dict) -> str:
    player_name = _s((digest_json or {}).get("player_name")) or "Verified Player"
    display_range = _s((digest_json or {}).get("display_range"))
    if display_range:
        return f"{player_name} verified update · {display_range}"
    return f"{player_name} verified update"


def build_player_update_email_html(digest_json: dict, chart_cid: str | None) -> str:
    digest = digest_json or {}
    summary = digest.get("summary") or {}
    badges = digest.get("badges_grouped") or digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    matches = digest.get("matches_played_rows") or []
    highlights = digest.get("highlights") or []
    glance = digest.get("week_at_a_glance") or []
    people = digest.get("people") or {}
    links = digest.get("links") or {}

    player_name = html.escape(_s(digest.get("player_name")) or "Verified Player")
    display_range = html.escape(_s(digest.get("display_range")))
    overall_text = _num(summary.get("overall_jupr_after"), digits=3)
    delta_text = _num(summary.get("overall_delta"), digits=4, prefix_sign=True, default="—")
    record = html.escape(_s(summary.get("record")) or "0-0")
    matches_played = int(summary.get("matches_played") or 0)

    highlight_items = "".join(
        f"<tr><td style='padding:4px 0;'><span style='color:#FF7A00;'>•</span> {html.escape(_s(line))}</td></tr>"
        for line in highlights[:6]
        if _s(line)
    )
    glance_items = "".join(
        f"<tr><td style='padding:3px 0;color:#314155;'>{html.escape(_s(line))}</td></tr>"
        for line in glance[:4]
        if _s(line)
    )

    badge_items = "".join(_badge_line(item) for item in badges[:10])
    trophy_items = "".join(
        f"<tr><td style='padding:3px 0;'>• {html.escape(_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy')}</td></tr>"
        for item in trophies[:8]
    )
    match_items = "".join(
        f"<tr><td style='padding:5px 0;font-size:13px;line-height:1.35;'>{html.escape(_s(item.get('summary')))}</td></tr>"
        for item in matches
        if _s(item.get("summary"))
    )

    profile_link = html.escape(_s(links.get("player_profile")))
    unsubscribe_link = html.escape(_s(links.get("unsubscribe")))

    chart_block = (
        f"<img alt='Overall JUPR trend' src='cid:{html.escape(chart_cid)}' style='display:block;width:100%;max-width:560px;height:auto;border:1px solid #D9E0EB;border-radius:10px;'/>"
        if chart_cid
        else "<div style='color:#5A6678;font-size:13px;'>Chart unavailable for this date range.</div>"
    )

    return f"""
<html>
  <body style="margin:0;padding:0;background:#F2F4F8;font-family:Arial,sans-serif;color:#172033;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#F2F4F8;padding:22px 10px;">
      <tr>
        <td align="center">
          <table role="presentation" width="640" cellspacing="0" cellpadding="0" style="width:640px;max-width:100%;background:#FFFFFF;border:1px solid #D9E0EB;border-radius:14px;overflow:hidden;">
            <tr>
              <td style="padding:24px;background:linear-gradient(135deg,#FFF3E6,#FFE3C9);border-bottom:1px solid #ECD7BF;">
                <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#935A22;font-weight:700;">Verified Player Update</div>
                <div style="font-size:28px;line-height:1.15;font-weight:700;color:#111A2B;margin-top:4px;">{player_name}</div>
                <div style="font-size:14px;color:#3E4E67;margin-top:4px;">{display_range}</div>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 24px 6px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="25%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">JUPR</div><div style="font-size:22px;font-weight:700;">{overall_text}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="23%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Record</div><div style="font-size:22px;font-weight:700;">{record}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="23%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Matches</div><div style="font-size:22px;font-weight:700;">{matches_played}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="23%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Delta</div><div style="font-size:22px;font-weight:700;">{delta_text}</div></td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr><td style="padding:16px 24px 0;"><div style="font-size:17px;font-weight:700;margin-bottom:8px;">Overall JUPR Trend</div>{chart_block}</td></tr>
            <tr><td style="padding:14px 24px 0;"><div style="font-size:17px;font-weight:700;">Short Summary</div><table role="presentation" width="100%">{glance_items or "<tr><td style='padding:4px 0;color:#5A6678;'>No activity in this date window.</td></tr>"}</table></td></tr>
            <tr>
              <td style="padding:18px 24px 0;">
                <div style="font-size:17px;font-weight:700;margin-bottom:6px;">Highlights</div>
                <table role="presentation" width="100%">{highlight_items or "<tr><td style='padding:4px 0;color:#5A6678;'>No highlights available.</td></tr>"}</table>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 24px 0;">
                <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Badges earned</div>
                <table role='presentation' width='100%'>{badge_items or "<tr><td style='padding:3px 0;color:#5A6678;'>None this period.</td></tr>"}</table>
              </td>
            </tr>
            <tr>
              <td style="padding:14px 24px 0;">
                <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Matches Played</div>
                <table role='presentation' width='100%'>{match_items or "<tr><td style='padding:3px 0;color:#5A6678;'>No matches in this date range.</td></tr>"}</table>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 24px 0;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0">
                  <tr>
                    <td width="50%" valign="top" style="padding-right:8px;">
                      <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Played With</div>
                      <table role="presentation" width="100%">{_people_html(people.get('top_partners') or [], 'No partner data in this period.')}</table>
                    </td>
                    <td width="50%" valign="top" style="padding-left:8px;">
                      <div style="font-size:16px;font-weight:700;margin-bottom:6px;">Faced Most</div>
                      <table role="presentation" width="100%">{_people_html(people.get('top_opponents') or [], 'No opponent data in this period.')}</table>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
            {"<tr><td style='padding:18px 24px 0;'><div style='font-size:16px;font-weight:700;margin-bottom:6px;'>Trophies</div><table role='presentation' width='100%'>" + trophy_items + "</table></td></tr>" if trophy_items else ""}
            <tr>
              <td align="center" style="padding:22px 24px 10px;">
                <a href="{profile_link}" style="display:inline-block;background:#14223A;color:#FFFFFF;text-decoration:none;padding:10px 18px;border-radius:8px;font-weight:700;font-size:14px;">View player profile</a>
              </td>
            </tr>
            <tr>
              <td style="padding:0 24px 20px;font-size:12px;color:#5A6678;text-align:center;">
                Receiving these because you're subscribed to verified player updates. <a href="{unsubscribe_link}" style="color:#1B4FA3;">Unsubscribe</a>.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
""".strip()


def build_player_update_email_text(digest_json: dict) -> str:
    digest = digest_json or {}
    summary = digest.get("summary") or {}
    badges = digest.get("badges_grouped") or digest.get("badges_earned") or []
    trophies = digest.get("trophies_earned") or []
    matches = digest.get("matches_played_rows") or []
    highlights = digest.get("highlights") or []
    glance = digest.get("week_at_a_glance") or []
    people = digest.get("people") or {}
    links = digest.get("links") or {}

    badge_lines = []
    for item in badges[:10]:
        name = _s(item.get("name")) or _s(item.get("badge_id")) or "Badge"
        count = int(item.get("count") or 1)
        label = f"{name} x{count}" if count > 1 else name
        badge_lines.append(f"- {label}: {_s(item.get('description')) or 'Award earned during this update window.'}")
    trophy_lines = [
        f"- {_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy'}"
        for item in trophies[:10]
    ]
    highlight_lines = [f"- {_s(line)}" for line in highlights[:10] if _s(line)]
    glance_lines = [f"- {_s(line)}" for line in glance[:6] if _s(line)]
    match_lines = [f"- {_s(item.get('summary'))}" for item in matches if _s(item.get("summary"))]

    partner_lines = [
        f"- {_s(item.get('player_name')) or ('Player #' + _s(item.get('player_id')))} · {int(item.get('matches') or 0)} matches · {_s(item.get('record')) or '0-0'}"
        for item in (people.get("top_partners") or [])[:3]
    ]
    opp_lines = [
        f"- {_s(item.get('player_name')) or ('Player #' + _s(item.get('player_id')))} · {int(item.get('matches') or 0)} matches · {_s(item.get('record')) or '0-0'}"
        for item in (people.get("top_opponents") or [])[:3]
    ]

    return "\n".join(
        [
            "Verified Player Update",
            "You’re receiving this email because you subscribed to JUPR player profile updates.",
            f"Player: {_s(digest.get('player_name'))}",
            f"Date range: {_s(digest.get('display_range'))}",
            f"Current overall JUPR: {_num(summary.get('overall_jupr_after'), digits=3)}",
            f"Weekly record: {_s(summary.get('record')) or '0-0'}",
            f"Matches played: {int(summary.get('matches_played') or 0)}",
            f"Overall delta: {_num(summary.get('overall_delta'), digits=4, prefix_sign=True, default='—')}",
            "",
            "Week at a glance:",
            *(glance_lines or ["- No activity in this date window."]),
            "",
            "Matches played:",
            *(match_lines or ["- No matches in this date window."]),
            "",
            "Highlights:",
            *(highlight_lines or ["- No highlights available."]),
            "",
            "Played with:",
            *(partner_lines or ["- No partner data in this period."]),
            "",
            "Faced most:",
            *(opp_lines or ["- No opponent data in this period."]),
            "",
            "Badges earned:",
            *(badge_lines or ["- None this period."]),
            *(
                ["", "Trophies earned:", *trophy_lines]
                if trophy_lines
                else []
            ),
            "",
            f"Profile link: {_s(links.get('player_profile'))}",
            f"Unsubscribe: {_s(links.get('unsubscribe'))}",
        ]
    ).strip()

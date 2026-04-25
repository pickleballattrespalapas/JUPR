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


def _truncate(text: str, limit: int = 80) -> str:
    value = _s(text)
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 1)].rstrip()}…"


def _badge_line(item: dict) -> str:
    name = _s(item.get("name")) or _s(item.get("badge_id")) or "Badge"
    count = int(item.get("count") or 1)
    label = f"{name} x{count}" if count > 1 else name
    desc = _truncate(_s(item.get("description")), limit=90)
    desc_block = (
        f"<div style='color:#5A6678;font-size:12px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{html.escape(desc)}</div>"
        if desc
        else ""
    )
    return (
        "<tr><td style='padding:4px 0;'>"
        f"<div><strong>{html.escape(label)}</strong></div>"
        f"{desc_block}"
        "</td></tr>"
    )


def build_player_update_email_subject(digest_json: dict) -> str:
    player_name = _s((digest_json or {}).get("player_name")) or "Verified Player"
    summary = (digest_json or {}).get("summary") or {}
    before_text = _num(summary.get("overall_jupr_before"), digits=3, default="")
    rating_text = _num(summary.get("overall_jupr_after"), digits=3, default="")
    delta_text = _num(summary.get("overall_delta"), digits=4, prefix_sign=True, default="")
    if before_text and rating_text:
        if delta_text:
            return f"{player_name} verified update: {before_text} → {rating_text} ({delta_text})"
        return f"{player_name} verified update: {before_text} → {rating_text}"
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
    links = digest.get("links") or {}

    player_name = html.escape(_s(digest.get("player_name")) or "Verified Player")
    display_range = html.escape(_s(digest.get("display_range")))
    before_text = _num(summary.get("overall_jupr_before"), digits=3, default="—")
    overall_text = _num(summary.get("overall_jupr_after"), digits=3, default="—")
    delta_text = _num(summary.get("overall_delta"), digits=4, prefix_sign=True, default="—")
    record = html.escape(_s(summary.get("record")) or "0-0")
    matches_played = int(summary.get("matches_played") or 0)
    prestige_badges = [item for item in badges if int(item.get("prestige") or 0) >= 50]

    badge_items = "".join(_badge_line(item) for item in prestige_badges[:3])
    trophy_items = "".join(
        f"<tr><td style='padding:3px 0;'>• {html.escape(_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy')}</td></tr>"
        for item in trophies[:3]
    )
    new_period_block = ""
    if badge_items or trophy_items:
        new_period_block = (
            "<tr><td style='padding:18px 24px 0;'>"
            "<div style='font-size:16px;font-weight:700;margin-bottom:6px;'>New this period</div>"
            + (f"<div style='font-size:14px;font-weight:700;margin-top:6px;'>Prestige badges</div><table role='presentation' width='100%'>{badge_items}</table>" if badge_items else "")
            + (f"<div style='font-size:14px;font-weight:700;margin-top:10px;'>Trophies</div><table role='presentation' width='100%'>{trophy_items}</table>" if trophy_items else "")
            + "</td></tr>"
        )
    match_items = "".join(
        f"<tr><td style='padding:2px 0;font-size:13px;line-height:1.25;'>{html.escape(_s(item.get('summary')))}</td></tr>"
        for item in matches
        if _s(item.get("summary"))
    )
    recent_matches_block = (
        "<tr><td style='padding:14px 24px 0;'>"
        "<div style='font-size:16px;font-weight:700;margin-bottom:6px;'>Recent matches</div>"
        f"<table role='presentation' width='100%'>{match_items}</table>"
        "</td></tr>"
        if match_items
        else ""
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
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="19%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Previous JUPR</div><div style="font-size:22px;font-weight:700;">{before_text}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="19%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">New JUPR</div><div style="font-size:22px;font-weight:700;">{overall_text}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="19%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Change</div><div style="font-size:22px;font-weight:700;">{delta_text}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="19%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Record</div><div style="font-size:22px;font-weight:700;">{record}</div></td>
                    <td width="2%"></td>
                    <td style="padding:8px;border-radius:10px;background:#F7FAFF;text-align:center;" width="19%"><div style="font-size:11px;color:#617189;text-transform:uppercase;">Matches</div><div style="font-size:22px;font-weight:700;">{matches_played}</div></td>
                  </tr>
                </table>
              </td>
            </tr>
            <tr><td style="padding:16px 24px 0;"><div style="font-size:17px;font-weight:700;margin-bottom:8px;">Overall JUPR Trend</div>{chart_block}</td></tr>
            {new_period_block}
            {recent_matches_block}
            <tr>
              <td align="center" style="padding:22px 24px 10px;">
                <a href="{profile_link}" style="display:inline-block;background:#14223A;color:#FFFFFF;text-decoration:none;padding:10px 18px;border-radius:8px;font-weight:700;font-size:14px;">View full player profile</a>
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
    links = digest.get("links") or {}
    prestige_badges = [item for item in badges if int(item.get("prestige") or 0) >= 50]

    badge_lines = []
    for item in prestige_badges[:3]:
        name = _s(item.get("name")) or _s(item.get("badge_id")) or "Badge"
        count = int(item.get("count") or 1)
        label = f"{name} x{count}" if count > 1 else name
        description = _truncate(_s(item.get("description")), limit=90)
        badge_lines.append(f"- {label}{': ' + description if description else ''}")
    trophy_lines = [
        f"- {_s(item.get('tournament_name')) or _s(item.get('league_name')) or 'Trophy'}"
        for item in trophies[:3]
    ]
    match_lines = [f"- {_s(item.get('summary'))}" for item in matches if _s(item.get("summary"))]

    return "\n".join(
        [
            "Verified Player Update",
            "You’re receiving this email because you subscribed to JUPR player profile updates.",
            f"Player: {_s(digest.get('player_name'))}",
            f"Date range: {_s(digest.get('display_range'))}",
            f"Before JUPR: {_num(summary.get('overall_jupr_before'), digits=3, default='—')}",
            f"New JUPR: {_num(summary.get('overall_jupr_after'), digits=3, default='—')}",
            f"Change: {_num(summary.get('overall_delta'), digits=4, prefix_sign=True, default='—')}",
            f"Record: {_s(summary.get('record')) or '0-0'}",
            f"Matches played: {int(summary.get('matches_played') or 0)}",
            "Rating trend graph: Available in the HTML version of this email and on your profile page.",
            "",
            *(["New this period:", "Prestige badges:", *badge_lines] if badge_lines else []),
            *(["Trophies:", *trophy_lines] if trophy_lines else []),
            *([""] if badge_lines or trophy_lines else []),
            *(["Recent matches:", *match_lines, ""] if match_lines else []),
            "",
            f"Profile link: {_s(links.get('player_profile'))}",
            f"Unsubscribe: {_s(links.get('unsubscribe'))}",
        ]
    ).strip()

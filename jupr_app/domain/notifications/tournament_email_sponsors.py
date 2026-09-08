"""Shared, email-safe sponsor presentation for every tournament message."""
from __future__ import annotations

import base64
import binascii
from html import escape
import re
from typing import Any
from urllib.parse import urlsplit

MAX_LOGO_BYTES = 64 * 1024
MAX_TOTAL_LOGO_BYTES = 256 * 1024
TIERS = ("presenting", "premier", "supporting")


def _website(value: Any) -> str:
    value = str(value or "").strip()
    try:
        parts = urlsplit(value)
        if (parts.scheme in {"http", "https"} and parts.hostname
                and not parts.username and not parts.password
                and not re.search(r"[\s\\\x00-\x1f]", value)):
            return value
    except ValueError:
        pass
    return ""


def _visible(sponsors: list[dict] | None) -> list[dict]:
    return sorted([row for row in (sponsors or [])[:50] if isinstance(row, dict)
                   and row.get("name") and row.get("is_visible") is not False],
                  key=lambda row: TIERS.index(row.get("tier")) if row.get("tier") in TIERS else 2)


def sponsor_inline_images(sponsors: list[dict] | None) -> dict[str, bytes]:
    images: dict[str, bytes] = {}
    total = 0
    for index, row in enumerate(_visible(sponsors)):
        encoded = row.get("logo_png_base64") or ""
        if not isinstance(encoded, str) or len(encoded) > MAX_LOGO_BYTES * 4 // 3 + 4:
            continue
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            continue
        if (not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > MAX_LOGO_BYTES
                or total + len(data) > MAX_TOTAL_LOGO_BYTES):
            continue
        images[f"tournament-sponsor-{index}"] = data
        total += len(data)
    return images


def _html_groups(sponsors: list[dict] | None) -> tuple[str, str]:
    rows = _visible(sponsors)
    images = sponsor_inline_images(rows)
    groups: dict[str, list[str]] = {tier: [] for tier in TIERS}
    credits: list[str] = []
    for index, row in enumerate(rows):
        tier = row.get("tier") if row.get("tier") in TIERS else "supporting"
        name = escape(str(row["name"]))
        website = _website(row.get("website"))
        linked_name = f'<a href="{escape(website, quote=True)}" style="color:#1e3a5f">{name}</a>' if website else name
        title = "Presented by " if tier == "presenting" else ""
        size = "18" if tier == "presenting" else "16"
        content = f'<p style="margin:0 0 8px;font-size:{size}px;font-weight:bold">{title}{linked_name}</p>'
        cid = f"tournament-sponsor-{index}"
        if tier == "presenting":
            credit = f'<td style="padding:12px 16px 12px 0;font-size:14px;line-height:1.4;vertical-align:middle">Presented by <strong>{linked_name}</strong></td>'
            if cid in images:
                credit += f'<td width="120" align="right" style="padding:12px 0;vertical-align:middle"><img src="cid:{cid}" alt="{name}" width="120" style="display:block;width:120px;max-width:100%;height:auto;border:0"></td>'
            credits.append(f'<tr>{credit}</tr>')
        if cid in images:
            content += f'<p style="margin:8px 0"><img src="cid:{cid}" alt="{name}" width="200" style="display:block;width:200px;max-width:100%;height:auto;border:0"></p>'
        for key in ("level", "public_description"):
            if row.get(key):
                content += '<p style="margin:6px 0;line-height:1.5">' + escape(str(row[key])).replace("\n", "<br>") + '</p>'
        if website:
            content += f'<p style="margin:6px 0;overflow-wrap:anywhere"><a href="{escape(website, quote=True)}">Visit {name}</a></p>'
        groups[tier].append(f'<tr><td style="padding:16px 0;font-size:14px;line-height:1.5">{content}</td></tr>')

    def table(content: str) -> str:
        return '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="width:100%;border-collapse:collapse">' + content + '</table>'

    header = '<div style="margin-bottom:24px;border-bottom:1px solid #e2e8f0">' + table("".join(credits)) + '</div>' if credits else ""
    footer = table("".join(groups["presenting"])) if groups["presenting"] else ""
    for tier, label in (("premier", "Supporting sponsors"), ("supporting", "Community sponsors")):
        if groups[tier]:
            footer += f'<h2 style="font-size:18px;margin:20px 0 0">{label}</h2>' + table("".join(groups[tier]))
    if footer:
        footer = '<div style="margin-top:32px;padding-top:8px;border-top:1px solid #e2e8f0">' + footer + '</div>'
    return header, footer


def with_sponsors_html(body: str, sponsors: list[dict] | None) -> str:
    header, footer = _html_groups(sponsors)
    if not header and not footer:
        return body
    if header:
        body = body.replace("</h1>", "</h1>" + header, 1)
    body = body.replace('<h1>', '<h1 style="font-size:24px;line-height:1.3;margin:0 0 12px">', 1)
    # Inline styles and a presentation table also work in email clients that
    # strip stylesheets. Leave the background unset for native dark-mode colors.
    container = '<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="width:100%;max-width:680px;border-collapse:collapse"><tr><td style="font-family:Arial,sans-serif;font-size:16px;line-height:1.6;padding:8px 0">'
    body = re.sub(r"(<body\b[^>]*>)", lambda match: match.group(1) + container, body, count=1)
    return body.replace("</body>", footer + "</td></tr></table></body>", 1)


def with_sponsors_text(body: str, sponsors: list[dict] | None) -> str:
    groups: dict[str, list[str]] = {tier: [] for tier in TIERS}
    credits: list[str] = []
    for row in _visible(sponsors):
        tier = row.get("tier") if row.get("tier") in TIERS else "supporting"
        lines = [str(row["name"])]
        if tier == "presenting":
            credits.append("Presented by " + str(row["name"]))
            lines[0] = "Presented by " + lines[0]
        lines.extend(str(row[key]) for key in ("level", "public_description") if row.get(key))
        if website := _website(row.get("website")):
            lines.append(website)
        groups[tier].append("\n".join(lines))
    header = "\n".join(credits)
    footer = "\n\n".join((label + "\n" if label else "") + "\n\n".join(groups[tier])
                         for tier, label in (("presenting", ""), ("premier", "Supporting sponsors"), ("supporting", "Community sponsors")) if groups[tier])
    return "\n\n".join(part for part in (header, body, footer) if part)


def sponsor_preview_html(body: str, sponsors: list[dict] | None) -> str:
    """The same email, with embedded PNGs viewable in a sandboxed browser frame."""
    for cid, data in sponsor_inline_images(sponsors).items():
        body = body.replace(f'"cid:{cid}"', '"data:image/png;base64,' + base64.b64encode(data).decode("ascii") + '"')
    return body

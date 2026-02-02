from __future__ import annotations

import html
import logging
import re
from markdown_it import MarkdownIt

from jupr_app.domain.gamification.badge_copy import BadgeCopyPlain

logger = logging.getLogger(__name__)

BADGE_RENDER_REV = "2026-02-02a"

_INLINE_MARKDOWN = MarkdownIt("commonmark", {"html": False})
_HTML_TAG_RE = re.compile(r"<[^>]*>")


def _tripwire_plain_text(text: str | None, *, field: str) -> str | None:
    if not text:
        return None
    raw_text = str(text)
    probe_text = raw_text
    if "&lt;" in raw_text or "&#60;" in raw_text or "&#x3c;" in raw_text:
        probe_text = html.unescape(raw_text)
    if "badge-card__" in probe_text or "<div" in probe_text or _HTML_TAG_RE.search(probe_text):
        logger.error("HTML detected at UI boundary for badge %s; stripping tags.", field)
        stripped = _HTML_TAG_RE.sub("", probe_text)
        stripped = stripped.replace("<", "").replace(">", "")
        return stripped.strip()
    return raw_text


def render_inline_badge_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _tripwire_plain_text(str(text), field="copy")
    escaped = html.escape(str(cleaned))
    return _INLINE_MARKDOWN.renderInline(escaped)


def render_badge_card_html(
    *,
    name: str,
    icon: str,
    copy_plain: BadgeCopyPlain,
    state_label: str | None = None,
) -> str:
    name_text = html.escape(str(name))
    icon_text = html.escape(str(icon))
    desc_html = render_inline_badge_text(_tripwire_plain_text(copy_plain.desc_text, field="desc_text"))
    req_html = render_inline_badge_text(_tripwire_plain_text(copy_plain.req_text, field="req_text"))
    meta_text = _tripwire_plain_text(copy_plain.meta_text, field="meta_text")
    meta_html = html.escape(str(meta_text)) if meta_text else ""
    state_html = html.escape(str(state_label)) if state_label else ""

    lines = [
        '<div class="badge-card">',
        f'<div class="badge-card__icon">{icon_text}</div>',
        f'<div class="badge-card__name" title="{name_text}">{name_text}</div>',
    ]
    if state_html:
        lines.append(f'<div class="badge-card__state">{state_html}</div>')
    if desc_html:
        lines.append(f'<div class="badge-card__desc">{desc_html}</div>')
    lines.append(f'<div class="badge-card__req"><span class="label">Req:</span> {req_html}</div>')
    if meta_html:
        lines.append(f'<div class="badge-card__meta">{meta_html}</div>')
    lines.append("</div>")
    return "\n".join(lines)

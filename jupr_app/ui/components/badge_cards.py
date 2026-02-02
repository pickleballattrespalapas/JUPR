from __future__ import annotations

import html

from markdown_it import MarkdownIt

from jupr_app.domain.gamification.badge_copy import BadgeCopyPlain

_INLINE_MARKDOWN = MarkdownIt("commonmark", {"html": False})


def render_inline_badge_text(text: str | None) -> str:
    if not text:
        return ""
    escaped = html.escape(str(text))
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
    desc_html = render_inline_badge_text(copy_plain.desc_text)
    req_html = render_inline_badge_text(copy_plain.req_text)
    meta_html = html.escape(str(copy_plain.meta_text)) if copy_plain.meta_text else ""
    state_html = html.escape(str(state_label)) if state_label else ""

    return f"""
    <div class="badge-card">
        <div class="badge-card__icon">{icon_text}</div>
        <div class="badge-card__name" title="{name_text}">{name_text}</div>
        {f'<div class="badge-card__state">{state_html}</div>' if state_html else ''}
        {f'<div class="badge-card__desc">{desc_html}</div>' if desc_html else ''}
        <div class="badge-card__req"><span class="label">Req:</span> {req_html}</div>
        <div class="badge-card__meta">{meta_html}</div>
    </div>
    """

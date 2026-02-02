from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]*>")


def _strip_html(text: str) -> str:
    stripped = _HTML_TAG_RE.sub("", text)
    return stripped.replace("<", "").replace(">", "").strip()


def _contains_html(text: str) -> bool:
    return "<" in text or "badge-card__" in text or "</" in text or "<div" in text


@dataclass(frozen=True)
class BadgeCopyPlain:
    desc_text: str | None
    req_text: str | None
    meta_text: str | None


def _normalize_requirement_text(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "Requirements TBD"
    sanitized = _sanitize_plain_text("req_text", text, context="normalize_requirement")
    cleaned = re.sub(r"^(requirements?)\s*:\s*", "", sanitized, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or "Requirements TBD"


def _normalize_plain_text(raw: Any, *, field: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    sanitized = _sanitize_plain_text(field, text, context="normalize_plain")
    return sanitized or None


def _sanitize_plain_text(field: str, text: str, *, context: str | None = None) -> str:
    if _contains_html(text):
        message = f"HTML detected in badge copy field {field}"
        if context:
            message = f"{message} ({context})"
        if __debug__:
            raise AssertionError(f"{message}: {text}")
        logger.error("%s; stripping tags.", message)
        return _strip_html(text)
    return text


def validate_badge_copy_plain(
    copy: BadgeCopyPlain,
    *,
    context: str | None = None,
) -> BadgeCopyPlain:
    return BadgeCopyPlain(
        desc_text=_sanitize_plain_text("desc_text", copy.desc_text, context=context)
        if copy.desc_text
        else None,
        req_text=_sanitize_plain_text("req_text", copy.req_text, context=context)
        if copy.req_text
        else None,
        meta_text=_sanitize_plain_text("meta_text", copy.meta_text, context=context)
        if copy.meta_text
        else None,
    )


def build_badge_copy_plain(
    badge: dict[str, Any],
    *,
    earners_count: int | None = None,
) -> BadgeCopyPlain:
    desc_text = _normalize_plain_text(badge.get("description_md"), field="desc_text")
    req_text = _normalize_requirement_text(badge.get("requirements"))
    meta_text = f"{earners_count} earners" if earners_count is not None else None

    copy = BadgeCopyPlain(desc_text=desc_text, req_text=req_text, meta_text=meta_text)
    return validate_badge_copy_plain(copy, context="build_badge_copy_plain")

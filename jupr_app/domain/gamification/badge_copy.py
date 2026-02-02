from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class BadgeCopyPlain:
    desc_text: str | None
    req_text: str | None
    meta_text: str | None


def _normalize_requirement_text(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return "Requirements TBD"
    cleaned = re.sub(r"^(requirements?)\s*:\s*", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or "Requirements TBD"


def _normalize_plain_text(raw: Any) -> str | None:
    text = str(raw or "").strip()
    return text or None


def _sanitize_plain_text(field: str, text: str) -> str:
    if "<" in text or ">" in text or "badge-card__" in text or "</" in text or "<div" in text:
        if __debug__:
            raise AssertionError(f"HTML detected in badge copy field {field}: {text}")
        stripped = _HTML_TAG_RE.sub("", text)
        stripped = stripped.replace("<", "").replace(">", "")
        logger.warning("HTML detected in badge copy field %s; stripping tags.", field)
        return stripped.strip()
    return text


def build_badge_copy_plain(
    badge: dict[str, Any],
    *,
    earners_count: int | None = None,
) -> BadgeCopyPlain:
    desc_text = _normalize_plain_text(badge.get("description_md"))
    req_text = _normalize_requirement_text(badge.get("requirements"))
    meta_text = f"{earners_count} earners" if earners_count is not None else None

    if desc_text:
        desc_text = _sanitize_plain_text("desc_text", desc_text)
    if req_text:
        req_text = _sanitize_plain_text("req_text", req_text)
    if meta_text:
        meta_text = _sanitize_plain_text("meta_text", meta_text)

    return BadgeCopyPlain(desc_text=desc_text, req_text=req_text, meta_text=meta_text)

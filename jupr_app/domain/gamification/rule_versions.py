from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS


def compute_badge_rule_version() -> str:
    payload = [asdict(badge) for badge in BADGE_DEFINITIONS]
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()
    return f"badge_catalog:{digest}"

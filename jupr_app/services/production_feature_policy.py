"""Explicit activation for the reviewed Tres Palapas operations release."""
from __future__ import annotations

import os
from urllib.parse import urlparse

FLAGS = {
    "badges": "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
    "generators": "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
}


def production_feature_enabled(feature: str, club_id: str = "tres_palapas") -> bool:
    """A staging flag or another club can never activate production writes."""
    return bool(
        feature in FLAGS
        and str(club_id) == "tres_palapas"
        and os.getenv("JUPR_ENV", "").strip().lower() == "production"
        and os.getenv("FLY_APP_NAME", "").strip() == "juprleagues-api"
        and urlparse(os.getenv("SUPABASE_URL", "")).hostname == "dnoockbwfenunhcibwfn.supabase.co"
        and os.getenv("JUPR_PRODUCTION_WRITE_POLICY", "").strip().lower() == "enabled"
        and os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip().lower() == "none"
        and os.getenv(FLAGS[feature], "").strip().lower() in {"1", "true", "yes", "on"}
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )

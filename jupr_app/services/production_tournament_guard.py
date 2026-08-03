from __future__ import annotations

import os

TRUTHY = {"1", "true", "yes", "y", "on"}
PRODUCTION_TOURNAMENT_WRITE_FLAG = "JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION"


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def production_tournament_writes_enabled() -> bool:
    return bool(
        os.getenv("JUPR_ENV", "").strip().lower() == "production"
        and os.getenv("JUPR_PRODUCTION_WRITE_POLICY", "").strip().lower() == "enabled"
        and _truthy(PRODUCTION_TOURNAMENT_WRITE_FLAG)
        and os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


def require_production_tournament_writes() -> None:
    if os.getenv("JUPR_ENV", "").strip().lower() != "production":
        return
    if os.getenv("JUPR_PRODUCTION_WRITE_POLICY", "").strip().lower() != "enabled":
        raise PermissionError("Production tournament writes are disabled by the production write policy.")
    if not _truthy(PRODUCTION_TOURNAMENT_WRITE_FLAG):
        raise PermissionError(f"Production tournament writes require {PRODUCTION_TOURNAMENT_WRITE_FLAG}=1.")
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise RuntimeError("Production tournament writes require the server-only Supabase service credential.")

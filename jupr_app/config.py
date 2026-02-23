import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


USE_BADGE_ENGINE_V3 = True
DEBUG_MODE = _env_flag("DEBUG_MODE", False)
PRODUCTION_MODE = _env_flag("PRODUCTION_MODE", False)

FEATURE_SESSION_LADDER = _env_flag("FEATURE_SESSION_LADDER", False)
FEATURE_LIVE_SCORING = _env_flag("FEATURE_LIVE_SCORING", False)
SESSION_LADDER_MIN_AWARD_SESSIONS = int(os.getenv("SESSION_LADDER_MIN_AWARD_SESSIONS", "4") or "4")

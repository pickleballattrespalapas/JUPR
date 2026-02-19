import os

USE_BADGE_ENGINE_V3 = True
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "true").lower() == "true"

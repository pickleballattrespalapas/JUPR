# jupr_app/ui/pages/__init__.py

from . import leaderboards
from . import match_explorer
from . import players
from . import match_uploader
from . import challenge_ladder
from . import challenge_ladder_admin
from . import faqs

# Restored pages
from . import league_manager
from . import match_log
from . import player_editor
from . import admin_tools
from . import admin_guide
from . import moneyball
from . import league_results

__all__ = [
    "leaderboards",
    "match_explorer",
    "players",
    "match_uploader",
    "challenge_ladder",
    "challenge_ladder_admin",
    "faqs",
    "league_manager",
    "match_log",
    "player_editor",
    "admin_tools",
    "admin_guide",
    "moneyball",
    "league_results",
]

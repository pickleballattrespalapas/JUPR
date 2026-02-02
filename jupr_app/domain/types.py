from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class AppContext:
    supabase: Any
    club_id: str
    df_players_all: pd.DataFrame
    df_players_active: pd.DataFrame
    df_leagues: pd.DataFrame
    df_matches: pd.DataFrame
    df_meta: pd.DataFrame
    df_badges: pd.DataFrame
    df_player_badges: pd.DataFrame
    name_to_id: dict
    id_to_name: dict
    public_mode: bool
    admin_logged_in: bool
    schema_degraded: bool = False
    schema_degraded_reason: str | None = None

# jupr_app/ui/context.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class AppContext:
    supabase: object
    club_id: str
    df_players_all: pd.DataFrame
    df_players_active: pd.DataFrame
    df_leagues: pd.DataFrame
    df_matches: pd.DataFrame
    df_meta: pd.DataFrame
    name_to_id: dict
    id_to_name: dict
    public_mode: bool
    admin_logged_in: bool

"""Lazy page proxies for routes that need compatibility guards."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

_PATCH_VERSION = "2026-07-01-trophy-dark-v3"
_CONTROL_EXCEPTION_NAMES = {"RerunException", "StopException"}
_OPTIONAL_LEAGUE_METADATA_COLUMNS = {
    "status",
    "started_at",
    "ended_at",
    "ended_by",
    "end_awards",
    "schedule_config",
    "court_board_defaults",
    "rules_config",
    "awards_config",
    "event_tags",
}
_LEGACY_LEAGUE_METADATA_COLUMNS = {"is_active", "min_games", "description", "k_factor"}
_TOP_PERFORMER_LABELS = {
    "highest_rating": "Highest Rating",
    "most_improved": "Most Improved",
    "best_win_pct": "Best Win %",
    "most_wins": "Most Wins",
}
_TOP_PERFORMER_BADGE_LABELS = {
    "top_performer_highest_rating": "Highest Rating",
    "top_performer_most_improved": "Most Improved",
    "top_performer_best_win_pct": "Best Win %",
    "top_performer_most_wins": "Most Wins",
}
_TOP_PERFORMER_PRESTIGE = {
    "highest_rating": 130,
    "most_improved": 125,
    "best_win_pct": 120,
    "most_wins": 115,
}
_BADGE_DEFINITION_COLUMNS = ["name", "prestige", "category", "rarity", "tier", "icon_key", "scope"]
_GENERIC_TROPHY_NAMES = {"", "badge", "trophy", "award"}


def _is_streamlit_control_exception(exc: BaseException) -> bool:
    return exc.__class__.__name__ in _CONTROL_EXCEPTION_NAMES


def _is_missingish(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        import pandas as pd

        result = pd.isna(value)
        return bool(result) if not hasattr(result, "any") else False
    except Exception:
        return False


def _clean_text(value: object) -> str:
    if _is_missingish(value):
        return ""
    return str(value).strip()


def _status_text(value: object) -> str:
    return _clean_text(value).lower()


def _is_archived_status(value: object) -> bool:
    return _status_text(value) == "archived"


def _is_ended_status(value: object) -> bool:
    return _status_text(value) in {"ended", "completed", "complete", "done"}


def _is_active_status(value: object) -> bool:
    return _status_text(value) in {"active", "running", "live"}


def _safe_json_dict(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# -----------------------------------------------------------------------------
# League Manager compatibility guard
# -----------------------------------------------------------------------------

def _prime_end_league_wizard_state(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    status = payload.get("status")
    if _is_archived_status(status):
        import streamlit as st

        st.session_state["end_league_wizard_open"] = False
        st.session_state["end_league_step"] = 1
        return
    if _is_ended_status(status) or payload.get("ended_at"):
        import streamlit as st

        if payload.get("ended_at"):
            st.session_state["end_league_frozen_at"] = str(payload.get("ended_at"))
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True


def _looks_like_optional_league_schema_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if not text:
        return False
    mentions_optional_column = any(col in text for col in _OPTIONAL_LEAGUE_METADATA_COLUMNS)
    mentions_schema = any(
        marker in text
        for marker in ["schema cache", "could not find", "does not exist", "column", "42703", "pgrst204"]
    )
    return mentions_optional_column and mentions_schema


def _minimal_lifecycle_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload[key] for key in ["status", "is_active", "started_at", "ended_at", "ended_by", "end_awards"] if key in payload}


def _legacy_league_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    status = payload.get("status")
    if _is_archived_status(status) or _is_ended_status(status):
        return {"is_active": False}
    allowed = {"is_active": True} if _is_active_status(status) else {}
    for key in _LEGACY_LEAGUE_METADATA_COLUMNS:
        if key in payload:
            allowed[key] = payload[key]
    if payload.get("is_active") is False:
        allowed["is_active"] = False
    return allowed


class _LeagueMetadataQueryGuard:
    def __init__(self, supabase: Any, inner: Any, payload: Any):
        self._supabase = supabase
        self._inner = inner
        self._payload = payload
        self._filters: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def eq(self, *args: Any, **kwargs: Any) -> "_LeagueMetadataQueryGuard":
        self._filters.append(("eq", args, kwargs))
        self._inner = self._inner.eq(*args, **kwargs)
        return self

    def _apply_filters(self, query: Any) -> Any:
        for method_name, method_args, method_kwargs in self._filters:
            query = getattr(query, method_name)(*method_args, **method_kwargs)
        return query

    def _retry_update(self, payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        query = self._supabase.table("leagues_metadata").update(payload)
        return self._apply_filters(query).execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._inner.execute(*args, **kwargs)
        except BaseException as exc:
            if _is_streamlit_control_exception(exc) or not _looks_like_optional_league_schema_error(exc):
                raise
            lifecycle_payload = _minimal_lifecycle_payload(self._payload)
            if lifecycle_payload:
                try:
                    return self._retry_update(lifecycle_payload, *args, **kwargs)
                except BaseException as lifecycle_exc:
                    if _is_streamlit_control_exception(lifecycle_exc) or not _looks_like_optional_league_schema_error(lifecycle_exc):
                        raise
            legacy_payload = _legacy_league_payload(self._payload)
            if not legacy_payload:
                raise
            return self._retry_update(legacy_payload, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _call(*args: Any, **kwargs: Any) -> Any:
            self._filters.append((name, args, kwargs))
            result = attr(*args, **kwargs)
            if result is not self._inner:
                self._inner = result
            return self

        return _call


class _LeagueMetadataTableGuard:
    def __init__(self, supabase: Any, inner: Any):
        self._supabase = supabase
        self._inner = inner

    def update(self, payload: Any, *args: Any, **kwargs: Any) -> _LeagueMetadataQueryGuard:
        _prime_end_league_wizard_state(payload)
        return _LeagueMetadataQueryGuard(self._supabase, self._inner.update(payload, *args, **kwargs), payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _LeagueManagerSupabaseGuard:
    def __init__(self, inner: Any):
        self._inner = inner

    def table(self, table_name: str) -> Any:
        table = self._inner.table(table_name)
        if str(table_name) == "leagues_metadata":
            return _LeagueMetadataTableGuard(self._inner, table)
        return table

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _maybe_advance_closed_league_wizard(ctx: Any) -> None:
    import pandas as pd
    import streamlit as st

    if not st.session_state.get("end_league_wizard_open", False):
        return
    try:
        current_step = int(st.session_state.get("end_league_step", 1) or 1)
    except Exception:
        current_step = 1
    if current_step > 1:
        return

    selected_league = _clean_text(st.session_state.get("league_editor_select", ""))
    if not selected_league:
        return
    df_meta = getattr(ctx, "df_meta", None)
    if df_meta is None or getattr(df_meta, "empty", True) or "league_name" not in df_meta.columns:
        return

    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    rows = meta[meta["league_name"] == selected_league]
    if rows.empty:
        return

    row = rows.iloc[0]
    ended_at = row.get("ended_at") if "ended_at" in rows.columns else None
    status = row.get("status") if "status" in rows.columns else ""
    has_ended_at = ended_at is not None and not pd.isna(ended_at) and _clean_text(ended_at) != ""

    if _is_archived_status(status):
        st.session_state["end_league_wizard_open"] = False
        st.session_state["end_league_step"] = 1
    elif _is_ended_status(status) or has_ended_at:
        if has_ended_at:
            st.session_state["end_league_frozen_at"] = str(ended_at)
        st.session_state["end_league_step"] = 2
        st.session_state["end_league_wizard_open"] = True


def _render_league_manager(ctx: Any) -> None:
    _maybe_advance_closed_league_wizard(ctx)
    module = importlib.import_module(f"{__name__}.league_manager")
    globals()["league_manager"] = _LEAGUE_MANAGER_PROXY
    original_supabase = getattr(ctx, "supabase", None)
    if original_supabase is not None:
        ctx.supabase = _LeagueManagerSupabaseGuard(original_supabase)
    try:
        module.render(ctx)
    finally:
        if original_supabase is not None:
            ctx.supabase = original_supabase


# -----------------------------------------------------------------------------
# Player Trophy Room patch
# -----------------------------------------------------------------------------

def _humanize_badge_id(badge_id: object) -> str:
    text = _clean_text(badge_id)
    if not text:
        return "Trophy"
    return " ".join(part for part in text.replace("-", "_").split("_") if part).title()


def _top_performer_category(row: Any) -> str:
    value_json = _safe_json_dict(row.get("value_json"))
    category_label = _clean_text(value_json.get("category_label") or value_json.get("category"))
    if category_label:
        return category_label
    category_key = _clean_text(value_json.get("category_key"))
    if category_key in _TOP_PERFORMER_LABELS:
        return _TOP_PERFORMER_LABELS[category_key]
    badge_id = _clean_text(row.get("badge_id"))
    if badge_id in _TOP_PERFORMER_BADGE_LABELS:
        return _TOP_PERFORMER_BADGE_LABELS[badge_id]
    if badge_id.startswith("top_performer_"):
        suffix = badge_id.removeprefix("top_performer_")
        return _TOP_PERFORMER_LABELS.get(suffix, _humanize_badge_id(suffix))
    context_id = _clean_text(row.get("context_id"))
    if ":top_performer:" in context_id:
        parts = context_id.split(":top_performer:", 1)[1].split(":")
        if parts:
            return _TOP_PERFORMER_LABELS.get(parts[0], _humanize_badge_id(parts[0]))
    return ""


def _is_top_performer_trophy_row(row: Any) -> bool:
    badge_id = _clean_text(row.get("badge_id"))
    value_json = _safe_json_dict(row.get("value_json"))
    context_id = _clean_text(row.get("context_id"))
    return (
        badge_id.startswith("top_performer_")
        or badge_id in _TOP_PERFORMER_BADGE_LABELS
        or _clean_text(value_json.get("category_key")) in _TOP_PERFORMER_LABELS
        or bool(_clean_text(value_json.get("category_label")))
        or ":top_performer:" in context_id
    )


def _top_performer_rank(value_json: dict[str, Any], context_id: str = "") -> str:
    rank = value_json.get("rank")
    if rank in (None, "") and ":top_performer:" in context_id:
        parts = context_id.split(":")
        if parts:
            rank = parts[-1]
    if rank in (None, ""):
        return ""
    try:
        return f"#{int(rank)}"
    except Exception:
        return f"#{rank}"


def _patched_trophy_display_name(row: Any) -> str:
    value_json = _safe_json_dict(row.get("value_json"))
    context_id = _clean_text(row.get("context_id"))
    if _is_top_performer_trophy_row(row):
        category = _top_performer_category(row) or "Top Performer"
        title = category if category.lower().startswith("top performer") else f"Top Performer: {category}"
        rank = _top_performer_rank(value_json, context_id)
        return f"{title} {rank}".strip()

    for key in ["badge_name", "name", "title"]:
        cleaned = _clean_text(row.get(key))
        if cleaned and cleaned.lower() not in _GENERIC_TROPHY_NAMES:
            return cleaned
    for key in ["tape_title", "award_name", "display_name", "title", "label", "category_label", "category"]:
        cleaned = _clean_text(value_json.get(key))
        if cleaned and cleaned.lower() not in _GENERIC_TROPHY_NAMES:
            return cleaned
    badge_id = _clean_text(row.get("badge_id"))
    if badge_id and badge_id.lower() not in _GENERIC_TROPHY_NAMES:
        return _humanize_badge_id(badge_id)
    return "Trophy"


def _missing_or_generic(value: object) -> bool:
    return _clean_text(value).lower() in _GENERIC_TROPHY_NAMES


def _merge_badge_definitions(df: Any, ctx: Any = None, supabase: Any = None) -> Any:
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "badge_id" not in df.columns:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    merged_df = df.copy()
    merged_df["badge_id"] = merged_df["badge_id"].fillna("").astype(str).str.strip()
    badge_ids = [bid for bid in merged_df["badge_id"].dropna().astype(str).unique().tolist() if bid]
    defs = getattr(ctx, "df_badges", None) if ctx is not None else None
    if defs is None or not isinstance(defs, pd.DataFrame) or defs.empty:
        defs = pd.DataFrame()
    if defs.empty and supabase is not None and badge_ids:
        try:
            resp = (
                supabase.table("badges")
                .select("badge_id,name,prestige,category,rarity,tier,icon_key,scope")
                .in_("badge_id", badge_ids)
                .execute()
            )
            defs = pd.DataFrame(resp.data or [])
        except Exception:
            defs = pd.DataFrame()

    if isinstance(defs, pd.DataFrame) and not defs.empty and "badge_id" in defs.columns:
        defs = defs.copy()
        defs["badge_id"] = defs["badge_id"].fillna("").astype(str).str.strip()
        keep_cols = ["badge_id"] + [c for c in _BADGE_DEFINITION_COLUMNS if c in defs.columns]
        defs = defs[keep_cols].drop_duplicates(subset=["badge_id"])
        joined = merged_df.merge(defs, on="badge_id", how="left", suffixes=("", "_def"))
        for col in _BADGE_DEFINITION_COLUMNS:
            def_col = f"{col}_def"
            if col not in joined.columns:
                joined[col] = pd.NA
            if def_col in joined.columns:
                mask = joined[col].isna() | (joined[col].astype(str).str.strip() == "")
                joined.loc[mask, col] = joined.loc[mask, def_col]
                joined = joined.drop(columns=[def_col])
        merged_df = joined

    if "name" not in merged_df.columns:
        merged_df["name"] = ""
    generic_name_mask = merged_df["name"].map(_missing_or_generic)
    if generic_name_mask.any():
        merged_df.loc[generic_name_mask, "name"] = merged_df.loc[generic_name_mask].apply(_patched_trophy_display_name, axis=1)
    if "prestige" not in merged_df.columns:
        merged_df["prestige"] = 0
    if "category" not in merged_df.columns:
        merged_df["category"] = pd.NA
    return merged_df


def _fetch_player_badges_resilient(supabase: Any, club_id: str, pid: int) -> Any:
    import pandas as pd

    select_cols = "id,club_id,player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json"
    try:
        resp = (
            supabase.table("player_badges")
            .select(select_cols)
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .execute()
        )
    except Exception:
        try:
            resp = (
                supabase.table("player_badges")
                .select("player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json")
                .eq("club_id", str(club_id))
                .eq("player_id", int(pid))
                .execute()
            )
        except Exception:
            return pd.DataFrame()
    pb_df = pd.DataFrame(resp.data or [])
    if pb_df.empty:
        return pb_df
    return _merge_badge_definitions(pb_df, supabase=supabase)


def _is_closed_league_row(row: Any) -> bool:
    import pandas as pd

    status = row.get("status") if hasattr(row, "get") else ""
    if _is_archived_status(status) or _is_ended_status(status):
        return True
    ended_at = row.get("ended_at") if hasattr(row, "get") else None
    if ended_at is not None and not pd.isna(ended_at) and _clean_text(ended_at):
        return True
    is_active = row.get("is_active") if hasattr(row, "get") else None
    if is_active is not None and not pd.isna(is_active) and not bool(is_active):
        return True
    return False


def _virtual_top_performer_badges(ctx: Any, club_id: str, pid: int) -> Any:
    import pandas as pd

    df_meta = getattr(ctx, "df_meta", None)
    df_leagues = getattr(ctx, "df_leagues", None)
    if df_meta is None or df_leagues is None or getattr(df_meta, "empty", True) or getattr(df_leagues, "empty", True):
        return pd.DataFrame()
    if "league_name" not in df_meta.columns:
        return pd.DataFrame()

    try:
        from jupr_app.domain.leagues import compute_top_performer_awards_for_config
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    id_to_name = getattr(ctx, "id_to_name", {}) or {}
    meta = df_meta.copy()
    meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
    closed_meta = meta[meta.apply(_is_closed_league_row, axis=1)].copy()
    if closed_meta.empty:
        return pd.DataFrame()

    for _, meta_row in closed_meta.iterrows():
        league_name = _clean_text(meta_row.get("league_name"))
        if not league_name:
            continue
        awards_config = _safe_json_dict(meta_row.get("awards_config"))
        try:
            awards = compute_top_performer_awards_for_config(
                df_leagues,
                df_meta,
                id_to_name,
                league_name,
                awards_config=awards_config,
            )
        except Exception:
            continue
        for award in awards or []:
            try:
                award_player_id = int(award.get("player_id"))
            except Exception:
                continue
            if award_player_id != int(pid):
                continue
            category_key = _clean_text(award.get("category_key"))
            category_label = _clean_text(award.get("category_label")) or _TOP_PERFORMER_LABELS.get(category_key, _humanize_badge_id(category_key))
            rank = award.get("rank") or 1
            badge_id = f"top_performer_{category_key}" if category_key else "top_performer_award"
            context_id = f"{league_name}:top_performer:{category_key}:{rank}"
            ended_at = meta_row.get("ended_at") if "ended_at" in meta_row.index else None
            earned_at = ended_at if ended_at is not None and not pd.isna(ended_at) else None
            value_json = {
                "league_id": league_name,
                "category_key": category_key,
                "category_label": category_label,
                "rank": rank,
                "metric_value": award.get("metric_value"),
                "metric_display": award.get("metric_display"),
                "ended_at": _clean_text(ended_at),
            }
            rows.append(
                {
                    "club_id": str(club_id),
                    "player_id": int(pid),
                    "badge_id": badge_id,
                    "earned_at": earned_at,
                    "context_type": "league",
                    "context_id": context_id,
                    "match_id": None,
                    "value_num": award.get("metric_value"),
                    "value_json": value_json,
                    "name": f"Top Performer: {category_label} #{rank}",
                    "prestige": _TOP_PERFORMER_PRESTIGE.get(category_key, 120),
                    "category": "Top Performer Awards",
                    "rarity": "legendary",
                    "icon_key": "trophy",
                    "scope": "league",
                }
            )
    return pd.DataFrame(rows)


def _dedupe_badges(df: Any) -> Any:
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    deduped = df.copy()
    subset = [col for col in ["player_id", "badge_id", "context_id"] if col in deduped.columns]
    if subset:
        deduped = deduped.drop_duplicates(subset=subset, keep="first")
    else:
        deduped = deduped.drop_duplicates()
    return deduped


def _theme_variables_css() -> str:
    try:
        import streamlit as st

        base = _clean_text(st.get_option("theme.base")).lower()
    except Exception:
        base = ""
    if base == "dark":
        return """
        :root{
          --text-primary:#f8fafc; --text-secondary:#cbd5e1; --text-muted:#cbd5e1;
          --panel:#111827; --border:#334155; --pill-bg:#1f2937;
          --shadow:none;
        }
        html,body{background:transparent;color:#f8fafc;}
        .trophy-case-card,.trophy-chip,.badge-card,.badge-stat{color:#f8fafc;}
        .trophy-case-meta,.trophy-body,.badge-subtext{color:#cbd5e1;}
        .trophy-case-header,.trophy-title,.badge-card-header{color:#f8fafc;}
        """
    return """
    :root{
      --text-primary:#111827; --text-secondary:#4b5563; --text-muted:#374151;
      --panel:#ffffff; --border:#e5e7eb; --pill-bg:#f8fafc;
      --shadow:none;
    }
    html,body{background:transparent;color:#111827;}
    """


def _patch_players_module(module: Any) -> Any:
    if getattr(module, "_JUPR_TROPHY_ROOM_PATCHED", "") == _PATCH_VERSION:
        return module

    original_resolve = getattr(module, "resolve_player_badges_for_profile", None)
    original_is_top_performer = getattr(module, "_is_top_performer_badge", None)
    original_st_html = getattr(module, "_JUPR_ORIGINAL_ST_HTML", None) or getattr(module, "st_html", None)
    module._JUPR_ORIGINAL_ST_HTML = original_st_html

    def patched_fetch_player_badges(supabase: Any, club_id: str, pid: int) -> Any:
        return _fetch_player_badges_resilient(supabase, club_id, pid)

    def patched_resolve_player_badges_for_profile(ctx: Any, supabase: Any, club_id: str, pid: int) -> Any:
        import pandas as pd

        frames = []
        virtual = _virtual_top_performer_badges(ctx, club_id, pid)
        if isinstance(virtual, pd.DataFrame) and not virtual.empty:
            frames.append(virtual)
        if callable(original_resolve):
            try:
                existing = original_resolve(ctx, supabase, club_id, pid)
                if isinstance(existing, pd.DataFrame) and not existing.empty:
                    frames.append(existing)
            except Exception:
                pass
        direct = _fetch_player_badges_resilient(supabase, club_id, pid)
        if isinstance(direct, pd.DataFrame) and not direct.empty:
            frames.append(direct)
        if not frames:
            return pd.DataFrame()
        combined = _dedupe_badges(pd.concat(frames, ignore_index=True, sort=False))
        return _merge_badge_definitions(combined, ctx=ctx, supabase=supabase)

    def patched_is_top_performer_badge(badge_id: object) -> bool:
        if callable(original_is_top_performer):
            try:
                if bool(original_is_top_performer(badge_id)):
                    return True
            except Exception:
                pass
        badge_key = _clean_text(badge_id)
        return badge_key.startswith("top_performer_") or badge_key in _TOP_PERFORMER_BADGE_LABELS

    def patched_st_html(html_block: str, *args: Any, **kwargs: Any) -> Any:
        if callable(original_st_html):
            theme_css = f"<style>{_theme_variables_css()}</style>"
            text = str(html_block or "")
            if "</head>" in text:
                text = text.replace("</head>", f"{theme_css}</head>", 1)
            else:
                text = f"{theme_css}{text}"
            return original_st_html(text, *args, **kwargs)
        return None

    module.fetch_player_badges = patched_fetch_player_badges
    module.resolve_player_badges_for_profile = patched_resolve_player_badges_for_profile
    module._trophy_display_name = _patched_trophy_display_name
    module._is_top_performer_badge = patched_is_top_performer_badge
    module.st_html = patched_st_html
    module._JUPR_TROPHY_ROOM_PATCHED = _PATCH_VERSION
    return module


def _render_players(ctx: Any) -> None:
    module = importlib.import_module(f"{__name__}.players")
    globals()["players"] = _PLAYERS_PROXY
    _patch_players_module(module)
    module.render(ctx)


_LEAGUE_MANAGER_PROXY = SimpleNamespace(render=_render_league_manager)
league_manager = _LEAGUE_MANAGER_PROXY
_PLAYERS_PROXY = SimpleNamespace(render=_render_players)
players = _PLAYERS_PROXY

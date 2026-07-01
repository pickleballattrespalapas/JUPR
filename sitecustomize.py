"""Runtime patch for Player Trophy Room rendering in Streamlit.

This module is loaded through ``jupr_app.__init__`` and installs an import hook so
it can patch ``jupr_app.ui.pages.players`` whenever Streamlit imports/reloads it.
"""
from __future__ import annotations

import builtins
import json
import sys
from typing import Any

_PATCH_VERSION = "2026-07-01-player-trophies-v5-wrap"
_ORIGINAL_IMPORT = builtins.__import__

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
_GENERIC_TROPHY_NAMES = {"", "badge", "trophy", "award"}
_BADGE_DEFINITION_COLUMNS = ["name", "prestige", "category", "rarity", "tier", "icon_key", "scope"]


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


def _humanize(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return "Trophy"
    return " ".join(part for part in text.replace("-", "_").split("_") if part).title()


def _top_performer_category(row: Any) -> str:
    value_json = _safe_json_dict(row.get("value_json"))
    label = _clean_text(value_json.get("category_label") or value_json.get("category"))
    if label:
        return label.removeprefix("Top Performer:").strip()
    key = _clean_text(value_json.get("category_key"))
    if key in _TOP_PERFORMER_LABELS:
        return _TOP_PERFORMER_LABELS[key]
    badge_id = _clean_text(row.get("badge_id"))
    if badge_id in _TOP_PERFORMER_BADGE_LABELS:
        return _TOP_PERFORMER_BADGE_LABELS[badge_id]
    if badge_id.startswith("top_performer_"):
        suffix = badge_id.removeprefix("top_performer_")
        return _TOP_PERFORMER_LABELS.get(suffix, _humanize(suffix))
    context_id = _clean_text(row.get("context_id"))
    if ":top_performer:" in context_id:
        parts = context_id.split(":top_performer:", 1)[1].split(":")
        if parts:
            return _TOP_PERFORMER_LABELS.get(parts[0], _humanize(parts[0]))
    return ""


def _is_top_performer_row(row: Any) -> bool:
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


def _top_performer_rank(row: Any) -> str:
    value_json = _safe_json_dict(row.get("value_json"))
    rank = value_json.get("rank")
    context_id = _clean_text(row.get("context_id"))
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


def _trophy_display_name(row: Any) -> str:
    if _is_top_performer_row(row):
        category = _top_performer_category(row) or "Top Performer"
        rank = _top_performer_rank(row)
        return f"{category} {rank}".strip()

    value_json = _safe_json_dict(row.get("value_json"))
    for key in ["badge_name", "name", "title"]:
        text = _clean_text(row.get(key))
        if text and text.lower() not in _GENERIC_TROPHY_NAMES:
            return text
    for key in ["tape_title", "award_name", "display_name", "label", "category_label", "category"]:
        text = _clean_text(value_json.get(key))
        if text and text.lower() not in _GENERIC_TROPHY_NAMES:
            return text
    badge_id = _clean_text(row.get("badge_id"))
    if badge_id and badge_id.lower() not in _GENERIC_TROPHY_NAMES:
        return _humanize(badge_id)
    return "Trophy"


def _format_top_performer_title(badge_name: str | None, category_label: str | None) -> str:
    """Use the category as the card title; rank/metric are shown below."""
    category = _clean_text(category_label).removeprefix("Top Performer:").strip()
    if category:
        return category
    name = _clean_text(badge_name).removeprefix("Top Performer:").strip()
    if name:
        # Strip a trailing rank if the patched display name already provided one.
        parts = name.rsplit(" #", 1)
        return parts[0].strip() if len(parts) == 2 and parts[1].isdigit() else name
    return "Top Performer"


def _missing_or_generic(value: object) -> bool:
    return _clean_text(value).lower() in _GENERIC_TROPHY_NAMES


def _merge_badge_definitions(df: Any, ctx: Any = None, supabase: Any = None) -> Any:
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "badge_id" not in df.columns:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    merged = df.copy()
    merged["badge_id"] = merged["badge_id"].fillna("").astype(str).str.strip()
    badge_ids = [bid for bid in merged["badge_id"].dropna().astype(str).unique().tolist() if bid]

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
        keep = ["badge_id"] + [col for col in _BADGE_DEFINITION_COLUMNS if col in defs.columns]
        defs = defs[keep].drop_duplicates(subset=["badge_id"])
        merged = merged.merge(defs, on="badge_id", how="left", suffixes=("", "_def"))
        for col in _BADGE_DEFINITION_COLUMNS:
            def_col = f"{col}_def"
            if col not in merged.columns:
                merged[col] = pd.NA
            if def_col in merged.columns:
                mask = merged[col].isna() | (merged[col].astype(str).str.strip() == "")
                merged.loc[mask, col] = merged.loc[mask, def_col]
                merged = merged.drop(columns=[def_col])

    if "name" not in merged.columns:
        merged["name"] = ""
    generic_name_mask = merged["name"].map(_missing_or_generic)
    if generic_name_mask.any():
        merged.loc[generic_name_mask, "name"] = merged.loc[generic_name_mask].apply(_trophy_display_name, axis=1)
    if "prestige" not in merged.columns:
        merged["prestige"] = 0
    if "category" not in merged.columns:
        merged["category"] = pd.NA
    return merged


def _fetch_player_badges(supabase: Any, club_id: str, pid: int) -> Any:
    import pandas as pd

    for select_cols in [
        "id,club_id,player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json",
        "player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json",
    ]:
        try:
            resp = (
                supabase.table("player_badges")
                .select(select_cols)
                .eq("club_id", str(club_id))
                .eq("player_id", int(pid))
                .execute()
            )
            df = pd.DataFrame(resp.data or [])
            return _merge_badge_definitions(df, supabase=supabase) if not df.empty else df
        except Exception:
            continue
    return pd.DataFrame()


def _is_closed_league_row(row: Any) -> bool:
    import pandas as pd

    status = _clean_text(row.get("status") if hasattr(row, "get") else "").lower()
    if status in {"archived", "ended", "completed", "complete", "done"}:
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
    for _, meta_row in closed_meta.iterrows():
        league_name = _clean_text(meta_row.get("league_name"))
        if not league_name:
            continue
        try:
            awards = compute_top_performer_awards_for_config(
                df_leagues,
                df_meta,
                id_to_name,
                league_name,
                awards_config=_safe_json_dict(meta_row.get("awards_config")),
            )
        except Exception:
            continue
        for award in awards or []:
            try:
                if int(award.get("player_id")) != int(pid):
                    continue
            except Exception:
                continue
            category_key = _clean_text(award.get("category_key"))
            category_label = _clean_text(award.get("category_label")) or _TOP_PERFORMER_LABELS.get(category_key, _humanize(category_key))
            rank = award.get("rank") or 1
            ended_at = meta_row.get("ended_at") if "ended_at" in meta_row.index else None
            badge_id = f"top_performer_{category_key}" if category_key else "top_performer_award"
            rows.append(
                {
                    "club_id": str(club_id),
                    "player_id": int(pid),
                    "badge_id": badge_id,
                    "earned_at": ended_at if ended_at is not None and not pd.isna(ended_at) else None,
                    "context_type": "league",
                    "context_id": f"{league_name}:top_performer:{category_key}:{rank}",
                    "match_id": None,
                    "value_num": award.get("metric_value"),
                    "value_json": {
                        "league_id": league_name,
                        "category_key": category_key,
                        "category_label": category_label,
                        "rank": rank,
                        "metric_value": award.get("metric_value"),
                        "metric_display": award.get("metric_display"),
                        "ended_at": _clean_text(ended_at),
                    },
                    "name": f"{category_label} #{rank}",
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
    subset = [col for col in ["player_id", "badge_id", "context_id"] if col in df.columns]
    return df.drop_duplicates(subset=subset, keep="first") if subset else df.drop_duplicates()


def _theme_css() -> str:
    try:
        import streamlit as st

        dark = _clean_text(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        dark = False
    base = """
    .trophy-case-grid{grid-template-columns:repeat(auto-fit,minmax(230px,1fr))!important;gap:1rem!important;}
    .trophy-case-card{min-height:132px!important;}
    .trophy-case-header{align-items:flex-start!important;gap:.55rem!important;}
    .trophy-case-header .truncate-1,
    .trophy-case-card .truncate-1{
      display:block!important;
      -webkit-line-clamp:unset!important;
      -webkit-box-orient:initial!important;
      overflow:visible!important;
      white-space:normal!important;
      text-overflow:clip!important;
      line-height:1.22!important;
    }
    .trophy-case-meta.truncate-1{white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
    """
    if dark:
        return base + """
        :root{--text-primary:#f8fafc;--text-secondary:#cbd5e1;--text-muted:#cbd5e1;--panel:#111827;--border:#334155;--pill-bg:#1f2937;--shadow:none;}
        html,body{background:transparent!important;color:#f8fafc!important;}
        .trophy-case-card,.trophy-chip,.badge-card,.badge-stat,.trophy-case-header,.trophy-title,.badge-card-header{color:#f8fafc!important;}
        .trophy-case-meta,.trophy-body,.badge-subtext{color:#cbd5e1!important;}
        """
    return base + """
    :root{--text-primary:#111827;--text-secondary:#4b5563;--text-muted:#374151;--panel:#ffffff;--border:#e5e7eb;--pill-bg:#f8fafc;--shadow:none;}
    html,body{background:transparent!important;color:#111827!important;}
    """


def _patch_players_module() -> None:
    module = sys.modules.get("jupr_app.ui.pages.players")
    if module is None:
        return
    if getattr(module, "_JUPR_SITE_PATCHED", "") == _PATCH_VERSION:
        return

    original_resolve = getattr(module, "resolve_player_badges_for_profile", None)
    original_is_top = getattr(module, "_is_top_performer_badge", None)
    original_st_html = getattr(module, "_JUPR_ORIGINAL_ST_HTML", None) or getattr(module, "st_html", None)
    module._JUPR_ORIGINAL_ST_HTML = original_st_html

    def patched_resolve(ctx: Any, supabase: Any, club_id: str, pid: int) -> Any:
        import pandas as pd

        frames = []
        for frame in [_virtual_top_performer_badges(ctx, club_id, pid)]:
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        if callable(original_resolve):
            try:
                existing = original_resolve(ctx, supabase, club_id, pid)
                if isinstance(existing, pd.DataFrame) and not existing.empty:
                    frames.append(existing)
            except Exception:
                pass
        direct = _fetch_player_badges(supabase, club_id, pid)
        if isinstance(direct, pd.DataFrame) and not direct.empty:
            frames.append(direct)
        if not frames:
            return pd.DataFrame()
        return _merge_badge_definitions(_dedupe_badges(pd.concat(frames, ignore_index=True, sort=False)), ctx=ctx, supabase=supabase)

    def patched_is_top(badge_id: object) -> bool:
        if callable(original_is_top):
            try:
                if bool(original_is_top(badge_id)):
                    return True
            except Exception:
                pass
        key = _clean_text(badge_id)
        return key.startswith("top_performer_") or key in _TOP_PERFORMER_BADGE_LABELS

    def patched_st_html(html_block: str, *args: Any, **kwargs: Any) -> Any:
        if not callable(original_st_html):
            return None
        css = f"<style>{_theme_css()}</style>"
        text = str(html_block or "")
        text = text.replace("</head>", f"{css}</head>", 1) if "</head>" in text else f"{css}{text}"
        return original_st_html(text, *args, **kwargs)

    module.fetch_player_badges = _fetch_player_badges
    module.resolve_player_badges_for_profile = patched_resolve
    module._trophy_display_name = _trophy_display_name
    module._format_top_performer_title = _format_top_performer_title
    module._is_top_performer_badge = patched_is_top
    module.st_html = patched_st_html
    module._JUPR_SITE_PATCHED = _PATCH_VERSION


def _import_hook(name: str, globals=None, locals=None, fromlist=(), level=0):
    result = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    try:
        if name == "jupr_app.ui.pages.players" or name == "jupr_app.ui.pages" or "players" in (fromlist or ()):
            _patch_players_module()
    except Exception:
        pass
    return result


builtins.__import__ = _import_hook
try:
    _patch_players_module()
except Exception:
    pass

from __future__ import annotations

import logging
import os
from dataclasses import replace
from math import isnan

import streamlit as st
import streamlit.components.v1 as components

from jupr_app.domain.gamification.badge_copy import build_badge_copy_plain
from jupr_app.domain.gamification.requirements import load_requirements_map, requirement_for
from jupr_app.ui.components import badge_cards
from jupr_app.ui.components.badge_cards import render_badge_card_html
from jupr_app.ui.pages.players import badge_icon

logger = logging.getLogger(__name__)

EARNS_PAGE_SIZE = 25

SECTION_ORDER = ["Common", "Uncommon", "Rare", "Legendary", "Unclaimed", "Unranked", "Other"]
CATALOG_SECTION_ORDER = ["Live Now", "Seasonal / League Close", "Manual / Curated", "Tracked / Disabled"]

CANONICAL_CATEGORY_LABELS = {
    "dominance": "Dominance",
    "consistency": "Consistency",
    "performance": "Performance",
    "activity": "Activity",
    "community": "Community",
    "streaks": "Streaks",
    "special": "Special",
}


def _badge_earner_counts(df_player_badges) -> dict[str, int]:
    if df_player_badges is None or df_player_badges.empty:
        return {}
    if "badge_id" not in df_player_badges.columns or "player_id" not in df_player_badges.columns:
        return {}
    unique = df_player_badges.drop_duplicates(subset=["badge_id", "player_id"])
    counts = unique["badge_id"].astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def get_all_badges(df_badges, df_player_badges, *, include_deprecated: bool = False) -> list[dict]:
    if df_badges is None or df_badges.empty:
        return []
    earners_map = _badge_earner_counts(df_player_badges)
    has_player_badges = df_player_badges is not None
    badges = []
    for row in df_badges.itertuples(index=False):
        badge_id = str(getattr(row, "badge_id", "") or "")
        name = str(getattr(row, "name", "") or "Badge")
        earners_count = earners_map.get(badge_id)
        if earners_count is None and has_player_badges:
            earners_count = 0
        state = _norm_badge_value(getattr(row, "state", None), default="live")
        badge_status = _norm_badge_value(getattr(row, "badge_status", None), default="live")
        badge_award_timing = _norm_badge_value(getattr(row, "badge_award_timing", None), default="live")
        badge_scope = _norm_optional_value(getattr(row, "badge_scope", None))
        if (
            state == "deprecated"
            and badge_status != "deprecated"
            and (earners_count is None or earners_count == 0)
        ):
            badge_status = "deprecated"
        if badge_status == "deprecated" and not include_deprecated and (earners_count is None or earners_count == 0):
            continue
        badges.append(
            {
                "badge_id": badge_id,
                "name": name,
                "category": _normalize_category_label(getattr(row, "category", None)),
                "prestige": getattr(row, "prestige", 0),
                "requirements": getattr(row, "requirements", None),
                "description_md": getattr(row, "description_md", None),
                "earners_count": earners_count,
                "state": state,
                "badge_status": badge_status,
                "badge_award_timing": badge_award_timing,
                "badge_scope": badge_scope,
            }
        )
    return sorted(badges, key=lambda item: item["name"].lower())


def _norm_badge_value(raw_value, *, default: str) -> str:
    if raw_value is None:
        return default
    if isinstance(raw_value, float) and isnan(raw_value):
        return default
    normalized = str(raw_value).strip().lower()
    if not normalized or normalized == "nan":
        return default
    return normalized


def _norm_optional_value(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and isnan(raw_value):
        return None
    normalized = str(raw_value).strip()
    if not normalized or normalized.lower() == "nan":
        return None
    return normalized


def _normalize_category_label(raw_category) -> str | None:
    if raw_category is None:
        return None
    label = str(raw_category).strip()
    if not label:
        return None
    key = " ".join(label.split()).lower()
    return CANONICAL_CATEGORY_LABELS.get(key, label.title())


def _badge_catalog_bucket(badge: dict) -> str:
    status = str(badge.get("badge_status") or "live").lower()
    timing = str(badge.get("badge_award_timing") or "live").lower()
    if status == "deprecated":
        return "Deprecated"
    if status in {"tracked", "disabled", "frozen"}:
        return "Tracked / Disabled"
    if timing in {"manual", "curated"}:
        return "Manual / Curated"
    if timing in {"on_league_close", "seasonal"}:
        return "Seasonal / League Close"
    return "Live Now"


def _availability_label(badge: dict) -> str:
    status = str(badge.get("badge_status") or "live").lower()
    timing = str(badge.get("badge_award_timing") or "live").lower()
    if status == "deprecated":
        return "Deprecated"
    if status == "disabled":
        return "Disabled"
    if status in {"tracked", "frozen"}:
        return "Tracked only"
    if timing == "manual":
        return "Manual award"
    if timing == "curated":
        return "Curated award"
    if timing == "on_league_close":
        return "Awarded on league close"
    if timing == "seasonal":
        return "Seasonal award"
    return "Live now"


def _show_earners_panel(badge: dict) -> bool:
    status = str(badge.get("badge_status") or "live").lower()
    return status in {"live", "deprecated"}


def _split_badges_for_catalog(badges: list[dict], *, include_deprecated: bool = False) -> dict[str, list[dict]]:
    grouped = {name: [] for name in CATALOG_SECTION_ORDER}
    if include_deprecated:
        grouped["Deprecated"] = []
    for badge in badges:
        bucket = _badge_catalog_bucket(badge)
        if bucket == "Deprecated":
            if include_deprecated:
                grouped["Deprecated"].append(badge)
            continue
        grouped.setdefault(bucket, []).append(badge)
    return grouped


def _group_badges(badges: list[dict]) -> list[tuple[str, list[dict]]]:
    has_category = any(badge.get("category") for badge in badges)
    sections: dict[str, list[dict]] = {}

    for badge in badges:
        if has_category:
            section = badge.get("category") or "Other"
        else:
            earners_count = badge.get("earners_count")
            if earners_count is None:
                section = "Unranked"
            elif earners_count == 0:
                section = "Unclaimed"
            elif earners_count >= 100:
                section = "Common"
            elif earners_count >= 25:
                section = "Uncommon"
            elif earners_count >= 5:
                section = "Rare"
            else:
                section = "Legendary"
        sections.setdefault(section, []).append(badge)

    if has_category:
        ordered_sections = sorted(sections.items(), key=lambda item: str(item[0]).lower())
    else:
        order_index = {name: idx for idx, name in enumerate(SECTION_ORDER)}
        ordered_sections = sorted(
            sections.items(),
            key=lambda item: (order_index.get(item[0], len(order_index)), str(item[0]).lower()),
        )

    return [(section, sorted(items, key=lambda item: item["name"].lower())) for section, items in ordered_sections]


def _render_badge_card(
    badge: dict,
    column,
    df_player_badges,
    df_players,
    *,
    debug_badges: bool = False,
) -> None:
    badge_id = badge.get("badge_id", "")
    name = badge.get("name", "Badge")
    icon = badge_icon(badge_id, badge.get("category"))
    earners_count = badge.get("earners_count")
    state_label = _availability_label(badge)
    status = str(badge.get("badge_status") or "live").lower()
    timing = str(badge.get("badge_award_timing") or "live").lower()
    scope = badge.get("badge_scope")
    meta_parts = []
    if scope:
        meta_parts.append(f"Scope: {scope}")
    meta_parts.append(state_label)
    if status == "live" and timing == "live":
        meta_parts.append("Currently obtainable")
    meta_line = " • ".join(meta_parts)
    open_key = f"badge_earners_open::{badge_id}"
    open_state = st.session_state.setdefault(open_key, False)
    toggle_label = (
        f"Earners ({earners_count}) {'▴' if open_state else '▾'}"
        if earners_count is not None
        else f"Earners {'▴' if open_state else '▾'}"
    )

    with column:
        copy_plain = build_badge_copy_plain(badge, earners_count=earners_count)
        copy_plain = replace(copy_plain, meta_text=meta_line)
        card_html = render_badge_card_html(
            name=name,
            icon=icon,
            copy_plain=copy_plain,
            state_label=None,
        )
        final_md = card_html
        if debug_badges and not st.session_state.get("badge_codex_debug_shown"):
            st.session_state["badge_codex_debug_shown"] = True
            lines = final_md.splitlines()
            has_blank_line = any(line.strip() == "" for line in lines)
            has_4space_lines = any(
                line.startswith("    ") or line.startswith("\t") for line in lines if line.strip()
            )
            blank_then_4space = any(
                lines[idx].strip() == "" and (lines[idx + 1].startswith("    ") or lines[idx + 1].startswith("\t"))
                for idx in range(len(lines) - 1)
            )
            first_nonblank = next((line for line in lines if line.strip()), "")
            leading_ws = len(first_nonblank) - len(first_nonblank.lstrip(" \t")) if first_nonblank else 0
            starts_with_div_at_col0 = first_nonblank.lstrip(" \t").startswith("<div") and leading_ws <= 3
            contains_escaped_lt = "&lt;" in final_md or "&#60;" in final_md or "&#x3c;" in final_md
            analysis_lines = ["line | lead_ws | blank | preview", "-" * 78]
            for idx, line in enumerate(lines, start=1):
                lead = len(line) - len(line.lstrip(" \t"))
                is_blank = line.strip() == ""
                analysis_lines.append(f"{idx:>4} | {lead:>7} | {str(is_blank):<5} | {line[:60]}")
            st.caption("Badge render debug (first card only).")
            st.text(f"badge_cards.__file__ = {badge_cards.__file__}")
            st.text(f"BADGE_RENDER_REV = {badge_cards.BADGE_RENDER_REV}")
            st.code(final_md, language="html")
            st.text("\n".join(analysis_lines))
            st.text(
                "Flags: "
                f"has_blank_line={has_blank_line}, "
                f"has_4space_lines={has_4space_lines}, "
                f"blank_then_4space={blank_then_4space}, "
                f"starts_with_div_at_col0={starts_with_div_at_col0}, "
                f"contains_escaped_lt={contains_escaped_lt}"
            )
            if blank_then_4space:
                st.warning("Debug guard: blank line followed by 4-space indentation detected.")
            components.html(final_md, height=260, scrolling=True)
        st.markdown(final_md, unsafe_allow_html=True)
        if _show_earners_panel(badge):
            if st.button(toggle_label, key=f"badge_codex_toggle_{badge_id}", use_container_width=True):
                open_state = not open_state
                st.session_state[open_key] = open_state

            if open_state:
                _render_earners_section(
                    badge_id,
                    earners_count,
                    df_player_badges,
                    df_players,
                )
        else:
            st.caption("Earners list hidden for non-obtainable badges.")


def get_badge_earners_page(
    df_player_badges,
    df_players,
    badge_id: str,
    offset: int,
    limit: int,
) -> tuple[list[dict], int]:
    if df_player_badges is None or df_player_badges.empty:
        return [], 0
    if "badge_id" not in df_player_badges.columns or "player_id" not in df_player_badges.columns:
        raise ValueError("Player badge data is missing required columns.")
    badge_id = str(badge_id)
    df_filtered = df_player_badges[df_player_badges["badge_id"].astype(str) == badge_id]
    if df_filtered.empty:
        return [], 0
    if "earned_at" in df_filtered.columns:
        df_filtered = df_filtered.sort_values("earned_at", ascending=False)
    df_filtered = df_filtered.drop_duplicates(subset=["player_id"])
    total = len(df_filtered)
    df_page = df_filtered.iloc[int(offset) : int(offset) + int(limit)]

    player_names = {}
    if df_players is not None and not df_players.empty and "id" in df_players.columns:
        names = df_players["name"] if "name" in df_players.columns else None
        if names is not None:
            player_names = dict(zip(df_players["id"], names.astype(str)))

    earners = []
    for row in df_page.itertuples(index=False):
        player_id = getattr(row, "player_id", None)
        name = player_names.get(player_id, f"Player {player_id}")
        earners.append({"player_id": player_id, "name": name})

    return earners, total


def _get_badge_cache(badge_id: str, total_known: int | None) -> dict:
    key = f"badge_earners_cache::{badge_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "players": [],
            "cursor": 0,
            "total": total_known,
            "has_more": total_known is None or total_known > 0,
            "loading": False,
            "error": None,
            "loaded_once": False,
        }
    return st.session_state[key]


def _load_more_earners(cache: dict, badge_id: str, df_player_badges, df_players) -> None:
    if cache["loading"] or cache.get("has_more") is False:
        return
    cache["loading"] = True
    try:
        new_earners, total = get_badge_earners_page(
            df_player_badges,
            df_players,
            badge_id,
            cache["cursor"],
            EARNS_PAGE_SIZE,
        )
        cache["players"].extend(new_earners)
        cache["cursor"] += len(new_earners)
        cache["total"] = total
        cache["has_more"] = cache["cursor"] < total
        cache["error"] = None
        cache["loaded_once"] = True
    except Exception as exc:  # noqa: BLE001 - surface per-badge errors
        cache["error"] = str(exc)
    finally:
        cache["loading"] = False


def _render_earners_section(badge_id: str, earners_count, df_player_badges, df_players) -> None:
    cache = _get_badge_cache(badge_id, earners_count)

    if earners_count == 0:
        st.caption("No one has earned this badge yet.")
        return

    if not cache["loaded_once"] and cache["error"] is None and not cache["loading"]:
        with st.spinner("Loading earners..."):
            _load_more_earners(cache, badge_id, df_player_badges, df_players)
        st.rerun()

    if cache["loading"]:
        st.info("Loading earners...")

    if cache["error"]:
        st.error(f"Couldn’t load earners. {cache['error']}")
        if st.button("Retry", key=f"badge_codex_retry_{badge_id}"):
            cache["error"] = None
            with st.spinner("Loading earners..."):
                _load_more_earners(cache, badge_id, df_player_badges, df_players)
            st.rerun()
        return

    if cache["loaded_once"]:
        if not cache["players"]:
            st.caption("No one has earned this badge yet.")
        else:
            if cache.get("total") is not None:
                st.caption(f"Earned by {cache['total']} players")
            for earner in cache["players"]:
                name = earner.get("name") or f"Player {earner.get('player_id', '')}"
                st.markdown(f"- 👤 {name}")

    if cache.get("has_more") and not cache["loading"]:
        if st.button("Load more", key=f"badge_codex_load_more_{badge_id}"):
            with st.spinner("Loading earners..."):
                _load_more_earners(cache, badge_id, df_player_badges, df_players)
            st.rerun()


def render(ctx) -> None:
    st.header("Badge Codex")
    st.caption("A full ledger of badges, with reels for the ones already on tape.")

    st.markdown(
        """
        <style>
            .badge-card {
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px;
                min-height: 170px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                background: var(--panel);
            }
            .badge-card__icon {
                font-size: 32px;
                line-height: 1;
            }
            .badge-card__name {
                font-weight: 600;
                font-size: 0.95rem;
                line-height: 1.2;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                text-overflow: ellipsis;
                min-height: 2.4em;
            }
            .badge-card__req {
                font-size: 0.8rem;
                color: var(--text-muted);
                white-space: normal;
                overflow: visible;
                text-overflow: clip;
                display: block;
            }
            .badge-card__req .label {
                font-weight: 600;
                margin-right: 0.25rem;
                color: var(--text-muted);
            }
            .badge-card__meta {
                font-size: 0.75rem;
                color: var(--text-muted);
                min-height: 1.2em;
            }
            .badge-card__state {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                font-size: 0.7rem;
                color: var(--text-muted);
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 0.1rem 0.5rem;
                align-self: flex-start;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_players = getattr(ctx, "df_players_all", None)
    debug_env = os.getenv("JUPR_DEBUG_BADGES") == "1"
    debug_badges = st.sidebar.checkbox("Debug badge render", value=debug_env)
    admin_mode = bool(getattr(ctx, "is_admin", False) or getattr(ctx, "admin_logged_in", False))
    debug_admin_mode = admin_mode or debug_env
    show_incomplete_audit = debug_env or bool(getattr(ctx, "is_admin", False) or getattr(ctx, "admin_logged_in", False))

    badge_defs = getattr(ctx, "df_badges", None)
    player_badges = getattr(ctx, "df_player_badges", None)
    if badge_defs is None:
        st.info("Badge data is still loading.")
        return

    include_deprecated = False
    if bool(getattr(ctx, "admin_logged_in", False)):
        include_deprecated = st.toggle("Include deprecated badges", value=False)

    badges = get_all_badges(badge_defs, player_badges, include_deprecated=include_deprecated)
    if not badges:
        st.caption("No badges are available yet.")
        return

    catalog_groups = _split_badges_for_catalog(badges, include_deprecated=include_deprecated)

    tabs = st.tabs(list(catalog_groups.keys()))
    for tab, tab_name in zip(tabs, catalog_groups):
        with tab:
            tab_badges = catalog_groups.get(tab_name, [])
            if not tab_badges:
                st.caption("No badges in this section.")
                continue
            for section_name, items in _group_badges(tab_badges):
                st.subheader(section_name)
                columns = st.columns(3)
                for idx, badge in enumerate(items):
                    _render_badge_card(
                        badge,
                        columns[idx % 3],
                        player_badges,
                        df_players,
                        debug_badges=debug_badges,
                    )
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if show_incomplete_audit:
        audit_rows = []
        for badge in badges:
            copy_plain = build_badge_copy_plain(badge, earners_count=badge.get("earners_count"))
            req_text = "" if copy_plain.req_text is None else str(copy_plain.req_text)
            is_incomplete = not req_text.strip() or "requirements tbd" in req_text.lower()
            if is_incomplete:
                audit_rows.append(
                    {
                        "badge_id": badge.get("badge_id", ""),
                        "name": badge.get("name", "Badge"),
                        "category": badge.get("category") or "",
                        "requirement_preview": req_text,
                    }
                )

        total_checked = len(badges)
        incomplete_count = len(audit_rows)
        logger.info(
            "Badge codex incomplete requirements audit: %s incomplete out of %s badges",
            incomplete_count,
            total_checked,
        )

        with st.expander("Admin: Incomplete Badges Audit", expanded=False):
            st.caption("Read-only audit: does not modify badges.")
            st.markdown(f"- Total badges checked: **{total_checked}**")
            st.markdown(f"- Incomplete badges: **{incomplete_count}**")
            if audit_rows:
                st.dataframe(audit_rows, use_container_width=True)
            else:
                st.success("All badges have requirement strings.")

            st.caption("Copy/paste stub pack for docs/badge_requirements.md")
            if audit_rows:
                stub_lines = []
                for entry in audit_rows:
                    badge_id = entry.get("badge_id", "")
                    name = entry.get("name", "Badge")
                    stub_lines.append(f"## {badge_id} — {name}")
                    stub_lines.append("Unlock: Requirements TBD.")
                    stub_lines.append("")
                st.code("\n".join(stub_lines).strip(), language="markdown")
            else:
                st.text("No stubs needed.")

        with st.expander("Admin: Requirement resolver debug", expanded=False):
            debug_badge_ids = [
                "above_expectations",
                "breakthrough",
                "dominant_run",
                "high_output",
            ]
            req_map = load_requirements_map()
            st.caption("Live resolver output for select badge IDs.")
            st.markdown(f"- Total requirements loaded: **{len(req_map)}**")
            for badge_id in debug_badge_ids:
                resolved = requirement_for(badge_id)
                exists = badge_id in req_map
                st.markdown(
                    f"- `{badge_id}`: exists={exists} · resolved: {resolved}",
                )

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import streamlit as st
from postgrest.exceptions import APIError

from jupr_app.domain.recaps.weekly_recap import (
    DEFAULT_SPOTLIGHT_DESCRIPTIONS,
    SPOTLIGHT_DEFAULT_ORDER,
    compute_weekly_recap,
    get_date_range_bounds,
    get_spotlight_candidates,
)
from jupr_app.ui.components.weekly_recap_layout import render_podium_layout, render_weekly_recap
from jupr_app.ui.layout import page_shell
from jupr_app.ui.url import qp_get




def render_tournament_podium(tournament: dict) -> None:
    st.subheader(str(tournament.get("tournament_name") or "Tournament"))
    st.markdown(render_podium_layout(tournament.get("podium", []) or []), unsafe_allow_html=True)


def _render_tournament_section(recap: dict) -> None:
    if "tournaments" not in recap:
        return
    st.header("🏆 Tournaments")
    tournaments = recap.get("tournaments") or []
    if not tournaments:
        st.info("Podium not available (missing tournament_podium rows)")
        return
    for tournament in tournaments:
        render_tournament_podium(tournament)

def _get_default_date_range(tz_name: str) -> tuple[date, date]:
    today = datetime.now(ZoneInfo(tz_name)).date()
    return today, today


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _handle_missing_table(exc: APIError) -> bool:
    code = _get_api_error_code(exc)
    if code in {"PGRST205", "42P01"}:
        st.error("Weekly recaps table not found. Apply migration migrations/20260207_weekly_recaps.sql in Supabase.")
        return True
    return False


def _load_weekly_row(supabase, club_id: str, start_date: date, end_date: date) -> dict | None:
    response = (
        supabase.table("weekly_recaps")
        .select("*")
        .eq("club_id", club_id)
        .eq("week_start", start_date.isoformat())
        .eq("week_end", end_date.isoformat())
        .execute()
    )
    if response.data:
        return response.data[0]
    return None


def _default_award(award_key: str, order: int) -> dict:
    return {
        "players": [],
        "description": DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(award_key, ""),
        "order": order,
        "include": True,
    }


def _normalize_overrides(overrides: dict, generated_spotlight: list[dict]) -> dict[str, dict]:
    normalized = {key: _default_award(key, idx + 1) for idx, key in enumerate(SPOTLIGHT_DEFAULT_ORDER)}
    for idx, item in enumerate(generated_spotlight or []):
        key = item.get("key")
        if key not in normalized:
            continue
        normalized[key] = {
            "players": list(item.get("candidate_ids") or item.get("players") or []),
            "description": item.get("description") or DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(key, ""),
            "order": int(item.get("order") or (idx + 1)),
            "include": bool(item.get("include", True)),
        }

    for key, value in (overrides or {}).items():
        if key not in normalized or not isinstance(value, dict):
            continue
        normalized[key]["players"] = list(value.get("players") or normalized[key]["players"])
        normalized[key]["description"] = value.get("description", normalized[key]["description"])
        normalized[key]["order"] = int(value.get("order") or normalized[key]["order"])
        normalized[key]["include"] = bool(value.get("include", normalized[key]["include"]))

    return normalized


def _load_debug_matches(supabase, club_id: str, start_dt_utc: datetime, end_dt_utc: datetime, limit: int = 50) -> list[dict]:
    columns = [
        "id",
        "date",
        "league",
        "match_type",
        "week_tag",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
        "score_t1",
        "score_t2",
    ]
    response = (
        supabase.table("matches")
        .select(",".join(columns))
        .eq("club_id", club_id)
        .gte("date", start_dt_utc.isoformat())
        .lte("date", end_dt_utc.isoformat())
        .order("date", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def _apply_edits(generated_json: dict, edits_json: dict, candidates: dict[str, list[dict]]) -> dict:
    recap = deepcopy(generated_json or {})
    if not recap:
        return recap

    looking_ahead = edits_json.get("looking_ahead")
    if looking_ahead:
        recap["looking_ahead"] = looking_ahead

    candidate_maps = {
        key: {item.get("candidate_id"): item for item in items if isinstance(item, dict)}
        for key, items in (candidates or {}).items()
    }

    generated_spotlight = recap.get("spotlight", []) or []
    overrides = _normalize_overrides(edits_json.get("spotlight_overrides", {}), generated_spotlight)

    updated = []
    for key, config in overrides.items():
        if not config.get("include", True):
            continue

        selected_ids = list(config.get("players") or [])[:3]
        selected_options = [candidate_maps.get(key, {}).get(candidate_id) for candidate_id in selected_ids]
        selected_options = [item for item in selected_options if item]

        if not selected_options:
            fallback = (candidates.get(key) or [])[:3]
            selected_options = [item for item in fallback if isinstance(item, dict)]

        if not selected_options:
            continue

        updated.append(
            {
                "key": key,
                "label": selected_options[0].get("label", key),
                "players": [item.get("display", "") for item in selected_options if item.get("display")],
                "candidate_ids": [item.get("candidate_id") for item in selected_options if item.get("candidate_id")],
                "description": config.get("description") or DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(key, ""),
                "order": int(config.get("order", 999)),
                "include": True,
            }
        )

    updated.sort(key=lambda item: int(item.get("order", 999)))
    recap["spotlight"] = updated
    return recap


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🗞️ Weekly Recap Admin", "Generate, edit, and publish the weekly recap.", mode_label=mode_label)

    print_mode = qp_get("print", "0").lower() in ("1", "true", "yes", "y")

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    tz_name = "America/Mazatlan"

    default_start, default_end = _get_default_date_range(tz_name)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    date_range_valid = bool(start_date and end_date and end_date >= start_date)
    if end_date < start_date:
        st.error("End date must be on or after start date.")
        date_range_valid = False
    day_span = (end_date - start_date).days + 1
    if day_span <= 0:
        st.error("Date range cannot be empty.")
        date_range_valid = False
    if day_span > 60:
        st.error("Date range cannot exceed 60 days.")
        date_range_valid = False

    with st.expander("Debug: date-range inputs", expanded=False):
        st.write(
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "day_span": day_span,
                "timezone": tz_name,
            }
        )

    try:
        row = _load_weekly_row(supabase, club_id, start_date, end_date)
    except APIError as exc:
        if _handle_missing_table(exc):
            return
        raise
    status = row.get("status") if row else "draft"

    st.write(f"Status: **{status}**")

    if st.button("Generate Draft", disabled=not date_range_valid):
        with st.spinner("Generating weekly recap..."):
            recap = compute_weekly_recap(ctx, start_date=start_date, end_date=end_date, include_tournaments=True, tz_name=tz_name)
            start_dt_utc, end_dt_utc = get_date_range_bounds(start_date, end_date, tz_name)

            with st.expander("Debug: recap match window", expanded=True):
                st.write(
                    {
                        "start_dt_utc": start_dt_utc.isoformat(),
                        "end_dt_utc": end_dt_utc.isoformat(),
                        "recap_matches": (recap.get("numbers") or {}).get("matches", 0),
                    }
                )
                try:
                    debug_rows = _load_debug_matches(supabase, club_id, start_dt_utc, end_dt_utc, limit=50)
                except APIError as exc:
                    st.warning(f"Unable to load debug matches: {exc}")
                else:
                    if debug_rows:
                        row_dates = [str(row.get("date")) for row in debug_rows if row.get("date")]
                        st.caption(f"Fetched {len(debug_rows)} rows (latest 50 by date).")
                        st.write(
                            {
                                "min_date": min(row_dates) if row_dates else None,
                                "max_date": max(row_dates) if row_dates else None,
                            }
                        )
                        st.dataframe(debug_rows, use_container_width=True, hide_index=True)
                    else:
                        st.info("No matches found within computed bounds.")

            payload = {
                "club_id": club_id,
                "week_start": start_date.isoformat(),
                "week_end": recap.get("end_date"),
                "status": "draft",
                "generated_json": recap,
                "edits_json": {},
                "final_json": recap,
            }
            try:
                supabase.table("weekly_recaps").upsert(payload, on_conflict="club_id,week_start").execute()
            except APIError as exc:
                if _handle_missing_table(exc):
                    return
                raise
            st.success("Draft generated.")
            st.rerun()

    if not row:
        st.info("No draft yet. Generate a draft to begin editing.")
        return

    generated_json = row.get("generated_json") or {}
    edits_json = row.get("edits_json") or {}
    candidates = get_spotlight_candidates(ctx, start_date=start_date, end_date=end_date, include_tournaments=True, tz_name=tz_name)

    st.subheader("Edit Draft")

    default_looking = edits_json.get("looking_ahead") or generated_json.get("looking_ahead") or ["", "", ""]
    looking_inputs = []
    for idx in range(3):
        looking_inputs.append(st.text_input(f"Looking Ahead #{idx + 1}", value=default_looking[idx] if idx < len(default_looking) else ""))

    edits_json["looking_ahead"] = looking_inputs

    st.markdown("**Spotlight Reel Overrides**")
    overrides = _normalize_overrides(edits_json.get("spotlight_overrides", {}), generated_json.get("spotlight", []))

    for idx, key in enumerate(SPOTLIGHT_DEFAULT_ORDER):
        options = candidates.get(key, []) or []
        if not options:
            continue

        label = options[0].get("label", key)
        option_labels = {item.get("candidate_id"): item.get("display", "") for item in options if item.get("candidate_id")}
        selected_default = [candidate_id for candidate_id in overrides[key].get("players", []) if candidate_id in option_labels]
        if not selected_default:
            selected_default = list(option_labels.keys())[:1]

        st.markdown(f"**{label}**")
        selected_players = st.multiselect(
            label,
            options=list(option_labels.keys()),
            default=selected_default[:3],
            max_selections=3,
            format_func=lambda opt: option_labels.get(opt, opt),
            key=f"{key}_players",
        )
        description = st.text_area(
            "Description / Explainer",
            value=overrides[key].get("description") or DEFAULT_SPOTLIGHT_DESCRIPTIONS.get(key, ""),
            key=f"{key}_description",
        )
        order = st.number_input(
            f"Order for {label}",
            min_value=1,
            max_value=10,
            value=int(overrides[key].get("order", idx + 1)),
            key=f"{key}_order",
        )
        include = st.checkbox(f"Include {label}", value=bool(overrides[key].get("include", True)), key=f"{key}_include")

        overrides[key] = {
            "players": selected_players,
            "description": description,
            "order": int(order),
            "include": include,
        }

    edits_json["spotlight_overrides"] = overrides

    if st.button("Save Draft"):
        final_json = _apply_edits(generated_json, edits_json, candidates)
        payload = {
            "club_id": club_id,
            "week_start": start_date.isoformat(),
            "week_end": generated_json.get("end_date"),
            "status": "draft",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": final_json,
        }
        try:
            supabase.table("weekly_recaps").upsert(payload, on_conflict="club_id,week_start").execute()
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Draft saved.")
        st.rerun()

    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    preview = st.checkbox("Preview (Print View)", value=print_mode)
    if preview:
        final_json = _apply_edits(generated_json, edits_json, candidates)
        st.caption("Tip: use browser print to PDF for a bulletin-ready copy.")
        render_weekly_recap(final_json, print_view=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Publish"):
        final_json = _apply_edits(generated_json, edits_json, candidates)
        payload = {
            "club_id": club_id,
            "week_start": start_date.isoformat(),
            "week_end": generated_json.get("end_date"),
            "status": "published",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": final_json,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "published_by": "admin",
        }
        try:
            supabase.table("weekly_recaps").upsert(payload, on_conflict="club_id,week_start").execute()
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Published.")
        st.rerun()

    if st.button("Unpublish"):
        payload = {
            "club_id": club_id,
            "week_start": start_date.isoformat(),
            "week_end": generated_json.get("end_date"),
            "status": "draft",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": _apply_edits(generated_json, edits_json, candidates),
            "published_at": None,
            "published_by": None,
        }
        try:
            supabase.table("weekly_recaps").upsert(payload, on_conflict="club_id,week_start").execute()
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Unpublished.")
        st.rerun()

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st
from postgrest.exceptions import APIError

from jupr_app.domain.recaps.weekly_recap import compute_weekly_recap, get_spotlight_candidates
from jupr_app.ui.components.weekly_recap_layout import render_weekly_recap
from jupr_app.ui.layout import page_shell
from jupr_app.ui.url import qp_get


def _get_default_date_range(tz_name: str) -> tuple[date, date]:
    today = datetime.now(ZoneInfo(tz_name)).date()
    current_week_start = today - timedelta(days=today.weekday())
    start_date = current_week_start - timedelta(days=7)
    end_date = current_week_start - timedelta(days=1)
    return start_date, end_date


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


def _apply_edits(generated_json: dict, edits_json: dict, candidates: dict[str, list[dict]]) -> dict:
    recap = deepcopy(generated_json or {})
    if not recap:
        return recap

    looking_ahead = edits_json.get("looking_ahead")
    if looking_ahead:
        recap["looking_ahead"] = looking_ahead

    spotlight = recap.get("spotlight", []) or []
    spotlight_by_key = {item.get("key"): item for item in spotlight}
    overrides = edits_json.get("spotlight_overrides", {})
    for key, candidate_id in overrides.items():
        options = {item.get("candidate_id"): item for item in candidates.get(key, [])}
        selected = options.get(candidate_id)
        if selected:
            spotlight_by_key[key] = selected

    dropped = set(edits_json.get("spotlight_drop", []))
    updated = [item for key, item in spotlight_by_key.items() if key not in dropped]

    order_map = edits_json.get("spotlight_order", {})
    if order_map:
        updated.sort(key=lambda item: order_map.get(item.get("key"), 999))

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
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    date_range_valid = bool(start_date and end_date and end_date >= start_date)
    if end_date < start_date:
        st.error("End date must be on or after start date.")

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
            recap = compute_weekly_recap(ctx, start_date=start_date, end_date=end_date, tz_name=tz_name)
            payload = {
                "club_id": club_id,
                "week_start": start_date.isoformat(),
                "week_end": recap.get("week_end"),
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
    candidates = get_spotlight_candidates(ctx, start_date=start_date, end_date=end_date, tz_name=tz_name)

    st.subheader("Edit Draft")

    default_looking = edits_json.get("looking_ahead") or generated_json.get("looking_ahead") or ["", "", ""]
    looking_inputs = []
    for idx in range(3):
        looking_inputs.append(st.text_input(f"Looking Ahead #{idx + 1}", value=default_looking[idx] if idx < len(default_looking) else ""))

    edits_json["looking_ahead"] = looking_inputs

    spotlight = generated_json.get("spotlight", [])
    spotlight_keys = [item.get("key") for item in spotlight if item.get("key")]

    st.markdown("**Spotlight Reel Overrides**")
    overrides = edits_json.get("spotlight_overrides", {})
    order_map = edits_json.get("spotlight_order", {})
    drop_list = set(edits_json.get("spotlight_drop", []))

    for idx, key in enumerate(spotlight_keys):
        options = candidates.get(key, [])
        if not options:
            continue
        label = options[0].get("label", key)
        option_labels = {item.get("candidate_id"): item.get("display", "") for item in options}
        current = overrides.get(key)
        if current not in option_labels:
            current = options[0].get("candidate_id")
        selection = st.selectbox(
            f"{label} candidate",
            options=list(option_labels.keys()),
            format_func=lambda opt: option_labels.get(opt, opt),
            index=list(option_labels.keys()).index(current) if current in option_labels else 0,
        )
        overrides[key] = selection
        order_map[key] = st.number_input(f"Order for {label}", min_value=1, max_value=10, value=int(order_map.get(key, idx + 1)))
        include = st.checkbox(f"Include {label}", value=key not in drop_list)
        if not include:
            drop_list.add(key)
        else:
            drop_list.discard(key)

    edits_json["spotlight_overrides"] = overrides
    edits_json["spotlight_order"] = order_map
    edits_json["spotlight_drop"] = list(drop_list)

    if st.button("Save Draft"):
        final_json = _apply_edits(generated_json, edits_json, candidates)
        payload = {
            "club_id": club_id,
            "week_start": start_date.isoformat(),
            "week_end": generated_json.get("week_end"),
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
            "week_end": generated_json.get("week_end"),
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
            "week_end": generated_json.get("week_end"),
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

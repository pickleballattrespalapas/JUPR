from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st
from postgrest.exceptions import APIError

from jupr_app.domain.recaps.weekly_recap import (
    DEFAULT_AWARD_DESCRIPTIONS,
    DEFAULT_AROUND_LEAGUE_DESCRIPTION,
    DEFAULT_AROUND_RR_DESCRIPTION,
    apply_around_descriptions,
    build_around_descriptions,
    compute_weekly_recap,
    get_spotlight_candidates,
)
from jupr_app.ui.components.weekly_recap_layout import build_weekly_recap_html, render_weekly_recap
from jupr_app.ui.layout import page_shell
from jupr_app.ui.url import qp_get


def _get_default_week_start(tz_name: str) -> date:
    today = datetime.now(ZoneInfo(tz_name)).date()
    return today - timedelta(days=today.weekday())


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


def _load_weekly_row(supabase, club_id: str, week_start: date) -> dict | None:
    response = (
        supabase.table("weekly_recaps")
        .select("*")
        .eq("club_id", club_id)
        .eq("week_start", week_start.isoformat())
        .execute()
    )
    if response.data:
        return response.data[0]
    return None


def _apply_edits(generated_json: dict, edits_json: dict, candidates: dict[str, list[dict]]) -> dict:
    recap = deepcopy(generated_json or {})
    if not recap:
        return recap

    if "print_theme" in edits_json:
        meta = recap.setdefault("meta", {})
        meta["print_theme"] = edits_json.get("print_theme")

    base_desc = generated_json.get("award_descriptions") or DEFAULT_AWARD_DESCRIPTIONS
    edited_desc = edits_json.get("award_descriptions") or {}
    merged_desc = dict(base_desc)
    merged_desc.update(edited_desc)

    recap["award_descriptions"] = merged_desc

    base_around_desc = generated_json.get("around_descriptions") or build_around_descriptions(
        recap.get("around_club") or {}
    )
    edited_around_desc = edits_json.get("around_descriptions") or {}
    merged_around_desc = dict(base_around_desc)
    merged_around_desc.update(edited_around_desc)
    recap["around_descriptions"] = merged_around_desc
    apply_around_descriptions(recap.get("around_club") or {}, merged_around_desc)

    looking_ahead = edits_json.get("looking_ahead")
    if looking_ahead:
        recap["looking_ahead"] = looking_ahead

    spotlight = recap.get("spotlight", []) or []
    spotlight_by_key = {item.get("key"): item for item in spotlight}
    overrides = edits_json.get("spotlight_overrides", {})
    spotlight_counts = edits_json.get("spotlight_counts", {})
    for key, candidate_id in overrides.items():
        if isinstance(candidate_id, str):
            candidate_ids = [candidate_id]
        else:
            candidate_ids = list(candidate_id or [])
        options = {item.get("candidate_id"): item for item in candidates.get(key, [])}
        selected_items = [options[candidate] for candidate in candidate_ids if candidate in options]
        if not selected_items:
            continue

        display_parts = [item.get("display", "") for item in selected_items]
        count_value = spotlight_counts.get(key)
        count_value = int(count_value) if count_value in {1, 2, 3} else len(display_parts)
        if count_value == 3:
            display_parts = [f"{idx + 1}) {display}" for idx, display in enumerate(display_parts)]
        if count_value >= 2:
            display = "<br/>".join(display_parts)
        else:
            display = display_parts[0] if display_parts else ""

        seen_player_ids = set()
        combined_player_ids = []
        for item in selected_items:
            for player_id in item.get("player_ids", []) or []:
                if player_id not in seen_player_ids:
                    combined_player_ids.append(player_id)
                    seen_player_ids.add(player_id)

        spotlight_by_key[key] = {
            "key": key,
            "label": selected_items[0].get("label"),
            "player_ids": combined_player_ids,
            "candidate_id": f"multi:{'|'.join(candidate_ids)}",
            "display": display,
        }

    dropped = set(edits_json.get("spotlight_drop", []))
    updated = [item for key, item in spotlight_by_key.items() if key not in dropped]

    order_map = edits_json.get("spotlight_order", {})
    if order_map:
        updated.sort(key=lambda item: order_map.get(item.get("key"), 999))

    for item in updated:
        item["description"] = merged_desc.get(item.get("key", ""), "")

    recap["spotlight"] = updated
    return recap


@st.cache_data(show_spinner=False)
def _pdf_for_recap(html: str) -> bytes:
    from weasyprint import HTML

    return HTML(string=html).write_pdf()


def _pdf_filename(final_json: dict) -> str:
    week_start = final_json.get("week_start") or "week_start"
    week_end = final_json.get("week_end") or "week_end"
    return f"weekly_recap_{week_start}_to_{week_end}.pdf"


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

    week_start = st.date_input("Week start (Monday)", value=_get_default_week_start(tz_name))
    allow_ties = st.checkbox("Allow ties (show 2 players when tied)", value=True)

    try:
        row = _load_weekly_row(supabase, club_id, week_start)
    except APIError as exc:
        if _handle_missing_table(exc):
            return
        raise
    status = row.get("status") if row else "draft"

    st.write(f"Status: **{status}**")

    if st.button("Generate Draft"):
        with st.spinner("Generating weekly recap..."):
            recap = compute_weekly_recap(ctx, week_start, tz_name=tz_name, allow_ties=allow_ties)
            payload = {
                "club_id": club_id,
                "week_start": week_start.isoformat(),
                "week_end": recap.get("week_end"),
                "status": "draft",
                "generated_json": recap,
                "edits_json": {},
                "final_json": recap,
            }
            try:
                sb_upsert(supabase, "weekly_recaps", payload, conflict="club_id,week_start")
            except APIError as exc:
                if _handle_missing_table(exc):
                    return
                raise
            st.success("Draft generated.")
            st.session_state["force_data_refresh"] = True

    if not row:
        st.info("No draft yet. Generate a draft to begin editing.")
        return

    generated_json = row.get("generated_json") or {}
    edits_json = row.get("edits_json") or {}
    candidates = get_spotlight_candidates(ctx, week_start, tz_name=tz_name, allow_ties=allow_ties)

    base_desc = generated_json.get("award_descriptions") or DEFAULT_AWARD_DESCRIPTIONS
    edited_desc = edits_json.get("award_descriptions") or {}
    merged_desc = dict(base_desc)
    merged_desc.update(edited_desc)

    base_around_desc = generated_json.get("around_descriptions") or build_around_descriptions(
        generated_json.get("around_club") or {}
    )
    edits_around_desc = edits_json.get("around_descriptions") or {}
    merged_around_desc = dict(base_around_desc)
    merged_around_desc.update(edits_around_desc)

    st.subheader("Edit Draft")

    default_looking = edits_json.get("looking_ahead") or generated_json.get("looking_ahead") or ["", "", ""]
    looking_inputs = []
    for idx in range(3):
        looking_inputs.append(st.text_input(f"Looking Ahead #{idx + 1}", value=default_looking[idx] if idx < len(default_looking) else ""))

    edits_json["looking_ahead"] = looking_inputs

    spotlight = generated_json.get("spotlight", [])
    spotlight_keys = [item.get("key") for item in spotlight if item.get("key")]
    spotlight_by_key = {item.get("key"): item for item in spotlight if item.get("key")}

    st.subheader("Award Descriptions")
    award_desc_edits = edits_json.get("award_descriptions") or {}
    for key in spotlight_keys:
        label = key
        options = candidates.get(key, [])
        if options:
            label = options[0].get("label", key)
        else:
            label = spotlight_by_key.get(key, {}).get("label", key)
        award_desc_edits[key] = st.text_area(
            f"{label} description",
            value=merged_desc.get(key, ""),
            height=120,
            key=f"award_desc_{key}",
        )
    edits_json["award_descriptions"] = award_desc_edits

    st.subheader("Around the Club Descriptions")
    around_desc_edits = edits_json.get("around_descriptions") or {}
    leagues = (generated_json.get("around_club") or {}).get("leagues") or []
    round_robins = (generated_json.get("around_club") or {}).get("round_robins") or []
    for league_item in leagues:
        league_name = str(league_item.get("league_name", "") or "").strip()
        if not league_name:
            continue
        key = f"LEAGUE:{league_name}"
        default_text = base_around_desc.get(key, DEFAULT_AROUND_LEAGUE_DESCRIPTION)
        around_desc_edits[key] = st.text_area(
            f"League: {league_name}",
            value=merged_around_desc.get(key, default_text),
            height=100,
            key=f"around_desc_{key}",
        )
    for rr_item in round_robins:
        event_id = str(rr_item.get("event_id", "") or "").strip()
        if not event_id:
            continue
        key = f"RR:{event_id}"
        label = rr_item.get("event_name", "Pop-Up Event") or "Pop-Up Event"
        default_text = base_around_desc.get(key, DEFAULT_AROUND_RR_DESCRIPTION)
        around_desc_edits[key] = st.text_area(
            f"Round Robin: {label}",
            value=merged_around_desc.get(key, default_text),
            height=100,
            key=f"around_desc_{key}",
        )
    edits_json["around_descriptions"] = around_desc_edits

    st.markdown("**Spotlight Reel Overrides**")
    overrides = edits_json.get("spotlight_overrides", {})
    spotlight_counts = edits_json.get("spotlight_counts", {})
    order_map = edits_json.get("spotlight_order", {})
    drop_list = set(edits_json.get("spotlight_drop", []))

    for idx, key in enumerate(spotlight_keys):
        options = candidates.get(key, [])
        if not options:
            continue
        label = options[0].get("label", key)
        option_ids = [item.get("candidate_id") for item in options if item.get("candidate_id")]
        option_labels = {item.get("candidate_id"): item.get("display", "") for item in options}
        current = overrides.get(key)
        if isinstance(current, str):
            current_list = [current]
        else:
            current_list = list(current or [])
        current_list = [candidate for candidate in current_list if candidate in option_labels]

        generated_item = spotlight_by_key.get(key, {})
        generated_player_ids = generated_item.get("player_ids") or []
        default_count = 2 if len(generated_player_ids) >= 2 else 1
        count_default = spotlight_counts.get(key, default_count)
        count_default = int(count_default) if count_default in {1, 2, 3} else default_count

        highlight_count = st.selectbox(
            f"{label} — How many to recognize",
            options=[1, 2, 3],
            index=[1, 2, 3].index(count_default),
        )
        spotlight_counts[key] = highlight_count

        if not current_list:
            if highlight_count in {2, 3}:
                current_list = option_ids[:highlight_count]
            else:
                generated_candidate_id = generated_item.get("candidate_id")
                if generated_candidate_id in option_labels:
                    current_list = [generated_candidate_id]
                elif option_ids:
                    current_list = [option_ids[0]]

        current_list = current_list[:highlight_count]
        selection = st.multiselect(
            f"{label} winners",
            options=option_ids,
            format_func=lambda opt: option_labels.get(opt, opt),
            default=current_list,
            max_selections=highlight_count,
        )
        selection_ordered = [candidate for candidate in option_ids if candidate in selection]
        if not selection_ordered:
            selection_ordered = current_list or option_ids[:highlight_count]
        overrides[key] = selection_ordered[:highlight_count]
        order_map[key] = st.number_input(f"Order for {label}", min_value=1, max_value=10, value=int(order_map.get(key, idx + 1)))
        include = st.checkbox(f"Include {label}", value=key not in drop_list)
        if not include:
            drop_list.add(key)
        else:
            drop_list.discard(key)

    edits_json["spotlight_overrides"] = overrides
    edits_json["spotlight_counts"] = spotlight_counts
    edits_json["spotlight_order"] = order_map
    edits_json["spotlight_drop"] = list(drop_list)

    if st.button("Save Draft"):
        final_json = _apply_edits(generated_json, edits_json, candidates)
        payload = {
            "club_id": club_id,
            "week_start": week_start.isoformat(),
            "week_end": generated_json.get("week_end"),
            "status": "draft",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": final_json,
        }
        try:
            sb_upsert(supabase, "weekly_recaps", payload, conflict="club_id,week_start")
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Draft saved.")
        st.session_state["force_data_refresh"] = True

    st.markdown("<div class='no-print'>", unsafe_allow_html=True)
    theme_options = {
        "Baja Flair V2 (Flyer)": "baja_v2",
        "September Newsletter": "newsletter_sep",
        "Classic (Neutral)": "classic",
    }
    current_theme = edits_json.get("print_theme") or "baja_v2"
    if current_theme not in theme_options.values():
        current_theme = "baja_v2"
    labels = list(theme_options.keys())
    values = list(theme_options.values())
    selected_label = st.selectbox("Bulletin style", options=labels, index=values.index(current_theme))
    edits_json["print_theme"] = theme_options[selected_label]
    preview = st.checkbox("Preview (Print View)", value=print_mode)
    final_json = _apply_edits(generated_json, edits_json, candidates)
    if preview:
        st.caption("Tip: use browser print to PDF for a bulletin-ready copy.")
        # Admin preview: render via iframe to avoid Streamlit escaping <style> and showing raw CSS.
        st.session_state["weekly_recap_render_mode"] = "iframe"
        try:
            render_weekly_recap(final_json, print_view=True)
        finally:
            # Clean up so public/print-friendly rendering keeps using direct DOM.
            st.session_state.pop("weekly_recap_render_mode", None)
    st.subheader("Download PDF")
    html = build_weekly_recap_html(final_json, print_view=True)
    pdf_bytes = _pdf_for_recap(html)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=_pdf_filename(final_json),
        mime="application/pdf",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Publish"):
        final_json = _apply_edits(generated_json, edits_json, candidates)
        payload = {
            "club_id": club_id,
            "week_start": week_start.isoformat(),
            "week_end": generated_json.get("week_end"),
            "status": "published",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": final_json,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "published_by": "admin",
        }
        try:
            sb_upsert(supabase, "weekly_recaps", payload, conflict="club_id,week_start")
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Published.")
        st.session_state["force_data_refresh"] = True

    if st.button("Unpublish"):
        payload = {
            "club_id": club_id,
            "week_start": week_start.isoformat(),
            "week_end": generated_json.get("week_end"),
            "status": "draft",
            "generated_json": generated_json,
            "edits_json": edits_json,
            "final_json": _apply_edits(generated_json, edits_json, candidates),
            "published_at": None,
            "published_by": None,
        }
        try:
            sb_upsert(supabase, "weekly_recaps", payload, conflict="club_id,week_start")
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        st.success("Unpublished.")
        st.session_state["force_data_refresh"] = True

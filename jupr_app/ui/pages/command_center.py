from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from jupr_app.domain.gamification.v3_engine import USE_BADGE_ENGINE_V3
from jupr_app.ui.components.theme_toggle import render_theme_toggle


_OPERATORS = (">=", ">", "=", "<=", "<", "is")


def _is_admin_user() -> bool:
    user = st.session_state.get("user")
    if isinstance(user, dict):
        return str(user.get("role") or "").strip().lower() == "admin"
    return False


def _badge_ui_enabled() -> bool:
    return bool(USE_BADGE_ENGINE_V3) and _is_admin_user()


def _build_condition_payload(row: dict) -> dict | None:
    fact_key = str(row.get("fact_key") or "").strip()
    operator = str(row.get("operator") or "").strip()
    value_type = str(row.get("value_type") or "").strip().lower()
    if not fact_key or operator not in _OPERATORS:
        return None

    payload = {
        "fact_key": fact_key,
        "operator": operator,
        "value_numeric": None,
        "value_boolean": None,
    }
    if operator == "is":
        payload["value_boolean"] = bool(row.get("value_boolean", False))
        return payload

    if value_type != "numeric":
        return None
    try:
        payload["value_numeric"] = float(row.get("value_numeric", 0))
    except (TypeError, ValueError):
        return None
    return payload


def _publish_badge(supabase, badge_id: str) -> None:
    supabase.table("badges").update(
        {
            "status": "published",
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("badge_id", badge_id).execute()


def _render_badge_admin_ui(ctx) -> None:
    st.subheader("Badge Engine V3")
    st.caption("Draft/publish controls for V3 badges.")

    try:
        badges = (
            ctx.supabase.table("badges")
            .select("badge_id,name,status,is_locked,award_count")
            .in_("status", ["draft", "published"])
            .order("status")
            .order("name")
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load badges: {exc}")
        badges = []

    if badges:
        st.markdown("#### Existing badges")
        for row in badges:
            badge_id = str(row.get("badge_id") or "")
            status = str(row.get("status") or "draft")
            is_locked = bool(row.get("is_locked", False))
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
                with c1:
                    st.markdown(f"**{row.get('name') or badge_id}**")
                    st.caption(badge_id)
                with c2:
                    st.write(f"Status: `{status}`")
                with c3:
                    st.write(f"Awards: {int(row.get('award_count') or 0)}")
                with c4:
                    st.write(f"Locked: {'Yes' if is_locked else 'No'}")
                if status == "draft":
                    if st.button("Publish", key=f"publish_badge_{badge_id}", disabled=is_locked):
                        try:
                            _publish_badge(ctx.supabase, badge_id)
                            st.success(f"Published {badge_id}.")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Publish failed: {exc}")
    else:
        st.info("No draft/published badges found.")

    st.markdown("#### Create draft badge")
    with st.form("badge_v3_create_form", clear_on_submit=True):
        badge_id = st.text_input("Badge ID", help="Unique key, e.g. grinder_v3")
        badge_name = st.text_input("Badge name")
        row_count = st.number_input("Condition rows", min_value=1, max_value=5, value=1, step=1)

        fact_rows = []
        fact_options = [""]
        try:
            facts = (
                ctx.supabase.table("badge_fact_registry")
                .select("fact_key,data_type")
                .order("fact_key")
                .execute()
                .data
                or []
            )
            fact_options += [str(f.get("fact_key") or "") for f in facts if f.get("fact_key")]
        except Exception:  # noqa: BLE001
            facts = []

        for idx in range(int(row_count)):
            c1, c2, c3 = st.columns([1.5, 1, 1])
            with c1:
                selected_fact = st.selectbox(
                    f"Fact #{idx + 1}",
                    options=fact_options,
                    key=f"badge_fact_{idx}",
                )
            with c2:
                operator = st.selectbox(f"Operator #{idx + 1}", options=_OPERATORS, key=f"badge_op_{idx}")
            with c3:
                if operator == "is":
                    bool_val = st.selectbox(f"Value #{idx + 1}", options=[True, False], key=f"badge_bool_{idx}")
                    fact_rows.append(
                        {
                            "fact_key": selected_fact,
                            "operator": operator,
                            "value_type": "boolean",
                            "value_boolean": bool_val,
                        }
                    )
                else:
                    num_val = st.number_input(f"Value #{idx + 1}", key=f"badge_num_{idx}", value=0.0)
                    fact_rows.append(
                        {
                            "fact_key": selected_fact,
                            "operator": operator,
                            "value_type": "numeric",
                            "value_numeric": num_val,
                        }
                    )

        submitted = st.form_submit_button("Create draft badge", use_container_width=True)
        if submitted:
            normalized_badge_id = str(badge_id or "").strip()
            if not normalized_badge_id or not badge_name.strip():
                st.error("Badge ID and name are required.")
                return

            payload_rows = []
            for fact_row in fact_rows:
                payload = _build_condition_payload(fact_row)
                if payload is None:
                    st.error("Each condition row must have a fact, operator, and valid value.")
                    return
                payload_rows.append(payload)

            try:
                ctx.supabase.table("badges").insert(
                    {
                        "badge_id": normalized_badge_id,
                        "name": badge_name.strip(),
                        "status": "draft",
                        "is_locked": False,
                        "award_count": 0,
                    }
                ).execute()
                condition_inserts = [{**row, "badge_id": normalized_badge_id} for row in payload_rows]
                if condition_inserts:
                    ctx.supabase.table("badge_rule_conditions").insert(condition_inserts).execute()
                st.success(f"Created draft badge {normalized_badge_id}.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to create draft badge: {exc}")


def render(ctx) -> None:
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    col_l, col_r = st.columns([0.8, 0.2])
    with col_l:
        st.title("JUPR Club Command Center")
        st.caption("Admin operations hub for match entry and league workflows.")
    with col_r:
        render_theme_toggle(key="cc_theme_toggle", label="Dark theme")

    st.info("Use the quick links below to move between common admin workflows.")

    st.subheader("Quick actions")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.link_button("Record Match", "/?page=record_match", use_container_width=True)
    with row1_col2:
        st.link_button("League Manager", "/?page=league_manager", use_container_width=True)
    with row1_col3:
        st.link_button("Match Log", "/?page=match_log", use_container_width=True)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.link_button("Player Editor", "/?page=player_editor", use_container_width=True)
    with row2_col2:
        st.link_button("Admin Tools", "/?page=admin_tools", use_container_width=True)
    with row2_col3:
        st.link_button("Weekly Recap Admin", "/?page=weekly_recap_admin", use_container_width=True)

    st.subheader("Alerts")
    st.info("Coming soon")

    st.subheader("Active competitions")
    st.info("Coming soon")

    st.subheader("Leaderboards snapshot")
    st.info("Coming soon")

    st.subheader("Public navigation")
    st.info("Coming soon")

    if _badge_ui_enabled():
        _render_badge_admin_ui(ctx)

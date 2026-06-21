from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.notifications.tournament_registration_confirmation_email import (
    PAYMENT_NOTE,
    build_registration_confirmation_view_model,
    format_money,
)
from jupr_app.domain.tournament_registration_repo import (
    get_public_tournament_bundle,
    get_registration_confirmation_bundle,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _partner_display(event: dict[str, Any]) -> str:
    mode = _safe_text(event.get("partner_mode")).upper()
    name = _safe_text(event.get("partner_name"))
    if mode == "HAS_PARTNER":
        return f"Partner: {name}" if name else "Partner entered"
    if mode == "NEEDS_PARTNER":
        return "Needs partner"
    return "—"


def _division_display(event: dict[str, Any]) -> str:
    parts = [_safe_text(event.get("division_name"))]
    skill = _safe_text(event.get("skill_label"))
    age = _safe_text(event.get("age_label"))
    if skill and skill.lower() != "open":
        parts.append(skill)
    if age and age.lower() not in {"all ages", "all"}:
        parts.append(age)
    return " • ".join(p for p in parts if p) or "Division"


def render(ctx) -> None:
    page_shell(
        "✅ Registration Confirmation",
        "Your tournament registration details and payment information.",
        mode_label="Public",
    )
    supabase = ctx.supabase
    available, reason = registration_feature_available(supabase)
    if not available:
        st.error(reason or "Tournament registration is not available.")
        return

    tournament_id = _safe_text(st.query_params.get("tournament_id"))
    slug = _safe_text(st.query_params.get("tournament"))
    registration_id = _safe_text(st.query_params.get("registration_id"))
    email_status = _safe_text(st.query_params.get("email_status"))
    if not tournament_id and slug:
        tournament, _settings, _days, _events = get_public_tournament_bundle(
            supabase, club_id=_safe_text(getattr(ctx, "club_id", "")), registration_slug=slug
        )
        tournament_id = _safe_text((tournament or {}).get("id"))
    if not tournament_id or not registration_id:
        st.error("We could not find that registration confirmation link.")
        return

    try:
        bundle = get_registration_confirmation_bundle(supabase, tournament_id, registration_id)
    except Exception:
        st.error("We could not load that registration confirmation right now. Please contact tournament staff.")
        return
    registration = bundle.get("registration") or {}
    if not registration:
        st.error("We could not find that registration.")
        return

    sender_status = get_smtp_config_status()
    vm = build_registration_confirmation_view_model(
        tournament=bundle.get("tournament"),
        registration=registration,
        selections=bundle.get("selections") or [],
        days=bundle.get("days") or [],
        event_options=bundle.get("event_options") or [],
        sender_from_name=sender_status.get("from_name"),
        sender_from_email=sender_status.get("from_email"),
    )

    st.title("Registration confirmed")
    st.success("Your tournament registration has been saved.")
    if email_status == "failed" or not sender_status.get("ok"):
        st.warning("Your registration was saved, but we could not send the confirmation email automatically. Tournament staff can still see your registration.")
    else:
        st.info("A confirmation email has been sent or should arrive shortly.")

    st.markdown(f"**Player/registrant:** {_safe_text(vm.get('display_name'))}")
    st.markdown(f"**Registered email:** {_safe_text(vm.get('email'))}")

    rows = [
        {
            "Day": _safe_text(event.get("day_label")),
            "Event": _safe_text(event.get("family_label")),
            "Division": _division_display(event),
            "Partner": _partner_display(event),
            "Price": format_money(event.get("price_usd")),
        }
        for event in vm.get("selected_events") or []
    ]
    if rows:
        st.table(rows)
    else:
        st.warning("No event selections were found for this registration.")
    st.subheader(f"Total price to pay: {format_money(vm.get('total_price_usd'))}")
    st.write(PAYMENT_NOTE)
    from_email = _safe_text(sender_status.get("from_email"))
    from_name = _safe_text(sender_status.get("from_name") or "JUPR Notifications")
    if from_email:
        st.caption(f"Your confirmation email will come from {from_name} <{from_email}>. Please check spam/junk if you do not see it.")
    else:
        st.caption("Your confirmation email will come from the tournament registration email address.")

    registration_slug = _safe_text((bundle.get("settings") or {}).get("registration_slug")) or slug
    nav_params = {"tournament_id": tournament_id}
    if registration_slug:
        nav_params["tournament"] = registration_slug
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Tournament Roster", key="confirmation_view_roster"):
            navigate_same_tab(page="tournament_roster", params=nav_params, public_mode=True)
    with col2:
        if st.button("Return to Tournament Registration", key="confirmation_return_registration"):
            navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)

from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.notifications.tournament_registration_confirmation_email import (
    PAYMENT_NOTE,
    build_registration_confirmation_view_model,
    format_money,
)


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


def render_registration_confirmation_summary(
    *,
    bundle: dict[str, Any],
    email_status: str,
    sender_status: dict[str, Any],
    show_title: bool = True,
) -> None:
    registration = bundle.get("registration") or {}
    vm = build_registration_confirmation_view_model(
        tournament=bundle.get("tournament"),
        registration=registration,
        selections=bundle.get("selections") or [],
        days=bundle.get("days") or [],
        event_options=bundle.get("event_options") or [],
        sender_from_name=sender_status.get("from_name"),
        sender_from_email=sender_status.get("from_email"),
    )

    if show_title:
        st.title("Registration confirmed")
    st.success("Your tournament registration has been saved.")
    if _safe_text(email_status) == "failed" or not sender_status.get("ok"):
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

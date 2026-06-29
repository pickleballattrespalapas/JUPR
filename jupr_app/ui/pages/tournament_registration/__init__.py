from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import streamlit as st

_LEGACY_MODULE_NAME = "jupr_app.ui.pages._tournament_registration_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tournament_registration.py"
_FLOW_CHOICE_KEY = "registration_flow_choice"
_FLOW_NEW = "new"
_FLOW_EDIT = "edit"


def _load_legacy_module():
    module = sys.modules.get(_LEGACY_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy tournament registration page from {_LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()
for _name in dir(_legacy):
    if _name == "render":
        continue
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = getattr(_legacy, _name)

_LEGACY_ADVANCE_STEP1 = _legacy._advance_step1_registration_wizard


def _reset_public_registration_wizard(tournament_id: str) -> dict[str, Any]:
    clear_registration_wizard_for_new_start(tournament_id)
    return _legacy._init_wizard_state(tournament_id)


def _registration_nav_params(*, tournament_id: str, registration_slug: str) -> dict[str, str]:
    params = {"tournament_id": tournament_id}
    if registration_slug:
        params["tournament"] = registration_slug
    return params


def _mask_registration_email(email: Any) -> str:
    try:
        return _legacy._mask_email(_safe_text(email))
    except Exception:
        return _safe_text(email)


def _send_edit_link(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    tournament_id: str,
    registration_id: str,
    email: str,
) -> tuple[bool, str]:
    try:
        token = build_registration_edit_token(
            tournament_id=tournament_id,
            registration_id=registration_id,
            email=email,
        )
        slug = _safe_text(settings.get("registration_slug"))
        edit_url = build_public_url(
            page="tournament_registration_edit",
            params={"tournament_id": tournament_id, "tournament": slug, "edit_token": token},
        )
        send_tournament_registration_edit_email(
            tournament_name=_safe_text(tournament.get("name") or "Tournament"),
            registered_email=email,
            edit_url=edit_url,
        )
        return True, ""
    except Exception as exc:
        error_text = str(exc).lower()
        if "configuration" in error_text or "jupr_registration_edit_secret" in error_text:
            return False, "Secure edit links are not configured yet. Please contact tournament staff to update your registration."
        return False, "We could not send the edit link automatically. Please contact tournament staff."


def _render_public_registration_choice(*, tournament_id: str, registration_slug: str) -> None:
    st.markdown("### Registration")
    st.write("Choose how you want to continue.")
    new_col, edit_col = st.columns(2)
    with new_col:
        if st.button("Start a new registration", type="primary", use_container_width=True, key=f"registration_start_new_{tournament_id}"):
            wizard = _reset_public_registration_wizard(tournament_id)
            wizard[_FLOW_CHOICE_KEY] = _FLOW_NEW
            wizard["current_step"] = 1
            wizard["edit_mode"] = False
            st.session_state[wizard_state_key(tournament_id)] = wizard
            navigate_same_tab(
                page="tournament_registration",
                params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
                public_mode=True,
                source="tournament_registration:start_new",
            )
    with edit_col:
        if st.button("Edit an existing registration", use_container_width=True, key=f"registration_start_edit_{tournament_id}"):
            wizard = _reset_public_registration_wizard(tournament_id)
            wizard[_FLOW_CHOICE_KEY] = _FLOW_EDIT
            wizard["current_step"] = 0
            st.session_state[wizard_state_key(tournament_id)] = wizard
            navigate_same_tab(
                page="tournament_registration",
                params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
                public_mode=True,
                source="tournament_registration:start_edit",
            )
    st.caption("If you already submitted a registration for this tournament, choose edit so we can email you a secure edit link.")


def _render_existing_registration_link_prompt(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    tournament_id: str,
    wizard: dict[str, Any],
) -> None:
    email = _safe_text(wizard.get("returning_email")).lower()
    registration_id = _safe_text(wizard.get("returning_registration_id"))
    masked = _mask_registration_email(email)
    st.markdown("### Edit existing registration")
    st.write(f"We found an existing registration for {masked}.")
    st.write("For your security, we’ll email a secure edit link to that address.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Email me a secure edit link", type="primary", use_container_width=True, key=f"edit_existing_send_{tournament_id}_{registration_id}"):
            sent, error = _send_edit_link(
                tournament=tournament,
                settings=settings,
                tournament_id=tournament_id,
                registration_id=registration_id,
                email=email,
            )
            wizard["returning_email_sent"] = sent
            wizard["returning_email_error"] = error
            st.session_state[wizard_state_key(tournament_id)] = wizard
            st.rerun()
    with c2:
        if st.button("Use a different email", use_container_width=True, key=f"edit_existing_different_{tournament_id}_{registration_id}"):
            wizard["returning_registration_id"] = ""
            wizard["returning_email"] = ""
            wizard["returning_email_sent"] = False
            wizard["returning_email_error"] = ""
            wizard[_FLOW_CHOICE_KEY] = _FLOW_EDIT
            st.session_state[wizard_state_key(tournament_id)] = wizard
            st.rerun()
    if wizard.get("returning_email_sent"):
        st.success(f"We sent an edit link to {masked}. Please check spam/junk.")
    if wizard.get("returning_email_error"):
        st.warning(_safe_text(wizard.get("returning_email_error")))


def _render_edit_lookup(
    *,
    supabase,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    tournament_id: str,
    registration_slug: str,
    wizard: dict[str, Any],
) -> None:
    if _safe_text(wizard.get("returning_registration_id")) and _safe_text(wizard.get("returning_email")):
        _render_existing_registration_link_prompt(
            tournament=tournament,
            settings=settings,
            tournament_id=tournament_id,
            wizard=wizard,
        )
        return

    st.markdown("### Edit existing registration")
    st.write("Enter the email address you used when you registered. We’ll send a secure edit link if we find a matching registration.")
    email = st.text_input("Registration email *", key=f"registration_edit_lookup_email_{tournament_id}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Send secure edit link", type="primary", use_container_width=True, key=f"registration_edit_lookup_submit_{tournament_id}"):
            clean_email = _safe_text(email).lower()
            if not clean_email:
                st.error("Enter the email address used for the registration.")
                st.stop()
            existing_registration = get_registration_by_email(supabase, tournament_id, clean_email)
            if not existing_registration:
                wizard["returning_email_error"] = "No registration was found for that email on this tournament. Check the email or start a new registration."
                wizard["returning_email_sent"] = False
                st.session_state[wizard_state_key(tournament_id)] = wizard
                st.rerun()
            wizard["returning_registration_id"] = _safe_text(existing_registration.get("id"))
            wizard["returning_email"] = clean_email
            sent, error = _send_edit_link(
                tournament=tournament,
                settings=settings,
                tournament_id=tournament_id,
                registration_id=_safe_text(existing_registration.get("id")),
                email=clean_email,
            )
            wizard["returning_email_sent"] = sent
            wizard["returning_email_error"] = error
            st.session_state[wizard_state_key(tournament_id)] = wizard
            st.rerun()
    with c2:
        if st.button("Back", use_container_width=True, key=f"registration_edit_lookup_back_{tournament_id}"):
            wizard = _reset_public_registration_wizard(tournament_id)
            st.session_state[wizard_state_key(tournament_id)] = wizard
            navigate_same_tab(
                page="tournament_registration",
                params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
                public_mode=True,
                source="tournament_registration:edit_back",
            )
    if wizard.get("returning_email_sent"):
        st.success(f"We sent an edit link to {_mask_registration_email(wizard.get('returning_email'))}. Please check spam/junk.")
    if wizard.get("returning_email_error"):
        st.warning(_safe_text(wizard.get("returning_email_error")))


def _advance_step1_registration_wizard(
    wizard: dict[str, Any],
    *,
    tournament_id: str,
    first_name: Any,
    last_name: Any,
    email_for_submit: Any,
    phone: Any,
    gender: Any,
    age: Any,
    notes: Any,
    find_existing_registration,
) -> tuple[bool, str]:
    if bool(wizard.get("edit_mode")) or _safe_text(wizard.get(_FLOW_CHOICE_KEY)) != _FLOW_NEW:
        return _LEGACY_ADVANCE_STEP1(
            wizard,
            tournament_id=tournament_id,
            first_name=first_name,
            last_name=last_name,
            email_for_submit=email_for_submit,
            phone=phone,
            gender=gender,
            age=age,
            notes=notes,
            find_existing_registration=find_existing_registration,
        )

    if not _safe_text(first_name) or not _safe_text(last_name) or not _safe_text(email_for_submit) or not _safe_text(age) or not _safe_text(gender):
        return False, "Please complete the highlighted required fields before continuing."

    normalized_email = _safe_text(email_for_submit).lower()
    wizard["step1"] = {
        "first_name": _safe_text(first_name),
        "last_name": _safe_text(last_name),
        "email": normalized_email,
        "phone": _safe_text(phone),
        "gender": _safe_text(gender),
        "age": _safe_text(age),
        "notes": _safe_text(notes),
    }
    existing_registration = find_existing_registration(tournament_id, normalized_email) if find_existing_registration else None
    if existing_registration:
        wizard[_FLOW_CHOICE_KEY] = _FLOW_EDIT
        wizard["returning_registration_id"] = _safe_text(existing_registration.get("id"))
        wizard["returning_email"] = normalized_email
        wizard["returning_email_sent"] = False
        wizard["returning_email_error"] = "That email already has a registration for this tournament. Use the secure edit link flow to make changes."
        wizard["current_step"] = 0
        return True, ""

    wizard["current_step"] = 2
    return True, ""


_legacy._advance_step1_registration_wizard = _advance_step1_registration_wizard
globals()["_advance_step1_registration_wizard"] = _advance_step1_registration_wizard


def _render_public_start_or_edit(ctx) -> bool:
    supabase = getattr(ctx, "supabase", None)
    club_id = _safe_text(getattr(ctx, "club_id", ""))
    if supabase is None or not club_id:
        st.error("Missing database context.")
        st.stop()

    available, detail = registration_feature_available(supabase)
    if not available:
        st.error("Tournament registration is not enabled yet. Apply the registration SQL migration first.")
        if detail:
            st.caption(detail)
        st.stop()

    tournament, settings, days, event_options = _legacy._select_public_tournament(ctx, supabase, page_key="tournament_registration")
    if not tournament:
        st.stop()

    tournament_id = _safe_text(tournament.get("id"))
    registration_slug = _safe_text(settings.get("registration_slug"))
    wizard = _legacy._init_wizard_state(tournament_id)
    flow_choice = _safe_text(wizard.get(_FLOW_CHOICE_KEY))

    if bool(wizard.get("edit_mode")) or _safe_text(st.query_params.get("edit")).lower() in {"1", "true", "yes", "y", "on"}:
        return False
    if flow_choice == _FLOW_NEW:
        return False

    page_shell(
        "📝 Tournament Registration",
        "Manage registration forms, player entries, approvals, partner needs, and public registration links.",
        mode_label="Public",
    )
    st.subheader(_safe_text(tournament.get("name") or "Tournament"))
    top_cols = st.columns([2, 1])
    with top_cols[0]:
        window_bits = []
        if settings.get("registration_open_at"):
            window_bits.append(f"Opens: {_safe_text(settings.get('registration_open_at'))}")
        if settings.get("registration_close_at"):
            window_bits.append(f"Closes: {_safe_text(settings.get('registration_close_at'))}")
        if window_bits:
            st.caption(" • ".join(window_bits))
    with top_cols[1]:
        if st.button("View Tournament Roster", key=f"start_view_tournament_roster_{tournament_id}"):
            navigate_same_tab(
                page="tournament_roster",
                params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
                public_mode=True,
                source="tournament_registration:start_view_roster",
            )

    enabled_days = [row for row in days if is_day_enabled(row)]
    selectable_event_options = [row for row in event_options if public_event_option_visibility(row) == "selectable"]
    hidden_draft_options = [row for row in event_options if _safe_text(row.get("status") or "draft").lower() == "draft"]
    is_open, _ = registration_is_open(settings)
    empty_message = _legacy._public_empty_state_message(
        registration_open=is_open,
        selectable_count=len(selectable_event_options),
        hidden_draft_count=len(hidden_draft_options),
    )
    if empty_message:
        st.warning(empty_message)
        st.stop()
    if not enabled_days or not selectable_event_options:
        st.warning("No open divisions are available right now.")
        st.stop()

    if flow_choice == _FLOW_EDIT:
        _render_edit_lookup(
            supabase=supabase,
            tournament=tournament,
            settings=settings,
            tournament_id=tournament_id,
            registration_slug=registration_slug,
            wizard=wizard,
        )
        return True

    _render_public_registration_choice(tournament_id=tournament_id, registration_slug=registration_slug)
    return True


def render(ctx) -> None:
    admin_mode = bool(getattr(ctx, "admin_logged_in", False)) and not bool(getattr(ctx, "public_mode", False))
    if not admin_mode and _render_public_start_or_edit(ctx):
        return
    _legacy.render(ctx)

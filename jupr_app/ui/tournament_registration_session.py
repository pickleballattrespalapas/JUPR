from __future__ import annotations

from typing import Any

import streamlit as st


def submission_state_key(tournament_id: Any) -> str:
    return f"registration_submission_result_{tournament_id}"


def wizard_state_key(tournament_id: Any) -> str:
    return f"registration_wizard_state_{tournament_id}"


def store_submission_result(
    *,
    tournament_id: str,
    registration_id: str,
    email_status: str,
    nav_params: dict[str, str],
) -> None:
    st.session_state[submission_state_key(tournament_id)] = {
        "registration_id": registration_id,
        "email_status": email_status,
        "nav_params": dict(nav_params),
    }


def get_submission_result(tournament_id: str) -> dict[str, Any]:
    return dict(st.session_state.get(submission_state_key(tournament_id), {}) or {})


def clear_registration_wizard_for_new_start(tournament_id: str) -> None:
    st.session_state.pop(wizard_state_key(tournament_id), None)
    st.session_state.pop(submission_state_key(tournament_id), None)

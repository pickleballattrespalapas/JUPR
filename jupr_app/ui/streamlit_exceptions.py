"""Helpers for preserving Streamlit control-flow exceptions."""

from __future__ import annotations


def is_streamlit_control_exception(exc: BaseException) -> bool:
    name = exc.__class__.__name__
    if name in {"RerunException", "StopException"}:
        return True

    module = getattr(exc.__class__, "__module__", "") or ""
    if "streamlit.runtime.scriptrunner" in module and (
        "Rerun" in name or "Stop" in name
    ):
        return True

    return False


def rethrow_if_streamlit_control(exc: BaseException) -> None:
    if is_streamlit_control_exception(exc):
        raise exc

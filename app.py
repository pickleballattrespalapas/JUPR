# jupr/app.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _sanitize_message(message: str) -> str:
    if not message:
        return ""

    redacted = message
    secret_patterns = [
        r"(?i)(api|anon|service|secret|token|password|key)[^\s'\"]{0,32}",
        r"[A-Za-z0-9_\-]{16,}",  # long tokens/keys
    ]
    for pattern in secret_patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


def _render_boot_error(exc: Exception) -> None:
    try:
        import streamlit as st

        st.set_page_config(page_title="JUPR Leagues – Boot Error", layout="wide")
        st.error("JUPR Leagues failed to start.")
        st.write("This is a startup error report with safe diagnostics.")

        st.markdown("**Exception type**")
        st.code(exc.__class__.__name__)

        st.markdown("**Exception message (sanitized)**")
        st.code(_sanitize_message(str(exc)))

        st.markdown("**Runtime diagnostics**")
        st.code(
            "\n".join(
                [
                    f"python_version: {sys.version.splitlines()[0]}",
                    f"cwd: {os.getcwd()}",
                    f"sys.path[:3]: {sys.path[:3]}",
                    f"listdir(cwd): {os.listdir(os.getcwd())[:30]}",
                ]
            )
        )
    except Exception:
        print("Failed to render Streamlit error page.")
        print(f"{exc.__class__.__name__}: {_sanitize_message(str(exc))}")


def _import_streamlit_app():
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))

    import importlib

    return importlib.import_module("streamlit_app")


try:
    streamlit_app = _import_streamlit_app()
    streamlit_app.main()
except Exception as e:
    _render_boot_error(e)
    raise

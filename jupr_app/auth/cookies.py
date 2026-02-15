from http.cookies import SimpleCookie

import streamlit.components.v1 as components

# NOTE:
# get_script_run_ctx is an internal Streamlit API and may change across versions.
# We guard the import for compatibility and pin Streamlit in requirements.txt.
try:
    # Modern Streamlit
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    # Legacy fallback
    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

COOKIE_NAME = "jupr_admin_session"


def set_cookie(value: str, max_age: int = 1209600):
    components.html(
        f"""
        <script>
        document.cookie = "{COOKIE_NAME}={value}; path=/; max-age={max_age}; SameSite=Lax";
        </script>
        """,
        height=0,
    )


def clear_cookie():
    set_cookie("", max_age=0)


def get_cookie():
    try:
        ctx = get_script_run_ctx()
        if not ctx or not ctx.request:
            return None

        cookie_header = ctx.request.headers.get("Cookie")
        if not cookie_header:
            return None

        cookie = SimpleCookie()
        cookie.load(cookie_header)

        if COOKIE_NAME in cookie:
            return cookie[COOKIE_NAME].value

    except Exception:
        return None

    return None

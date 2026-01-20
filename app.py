# jupr/app.py
from pathlib import Path
import sys
import streamlit as st
import traceback

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

st.write("Booting JUPR…")

try:
    import streamlit_app  # must be jupr/streamlit_app.py
    st.write("Imported module:", getattr(streamlit_app, "__file__", "unknown"))

    # Show candidate entrypoints
    candidates = ["main", "render", "run", "app"]
    found = [name for name in candidates if callable(getattr(streamlit_app, name, None))]
    st.write("Callable entrypoints found:", found)

    # Call the best available entrypoint
    if callable(getattr(streamlit_app, "main", None)):
        streamlit_app.main()
    elif callable(getattr(streamlit_app, "render", None)):
        streamlit_app.render()
    elif callable(getattr(streamlit_app, "run", None)):
        streamlit_app.run()
    elif callable(getattr(streamlit_app, "app", None)):
        streamlit_app.app()
    else:
        st.warning(
            "No entrypoint function found. "
            "Your streamlit_app.py must either execute Streamlit UI at import-time, "
            "or expose a function like main() or render()."
        )
except Exception:
    st.error("Crash while importing/running streamlit_app.py")
    st.code(traceback.format_exc())

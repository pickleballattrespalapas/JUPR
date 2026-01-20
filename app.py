# jupr/app.py
from pathlib import Path
import sys
import streamlit as st

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

st.write("Booting JUPR…")

import streamlit_app  # noqa: F401
if hasattr(streamlit_app, "main"):
    streamlit_app.main()

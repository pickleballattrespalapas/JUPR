# jupr/app.py
from pathlib import Path
import sys

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

import streamlit_app
streamlit_app.main()

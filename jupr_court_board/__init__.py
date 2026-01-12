import os
import streamlit.components.v1 as components
from typing import Any, Dict, List, Optional

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "build")
    _component = components.declare_component("jupr_court_board", path=build_dir)
else:
    _component = components.declare_component("jupr_court_board", url="http://localhost:5173")

def court_board(courts: List[Dict[str, Any]], key: Optional[str] = None) -> Dict[str, Any]:
    return _component(courts=courts, key=key, default={"courts": courts})

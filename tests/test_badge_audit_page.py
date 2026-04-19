from __future__ import annotations

import ast
from pathlib import Path

from jupr_app.ui.pages import badge_audit


def test_badge_audit_form_and_state_keys_do_not_collide():
    assert badge_audit.BADGE_AUDIT_FILTERS_FORM_KEY != badge_audit.BADGE_AUDIT_FILTERS_STATE_KEY


def test_badge_audit_repair_call_does_not_pass_include_non_live():
    source = Path("jupr_app/ui/pages/badge_audit.py").read_text()
    tree = ast.parse(source)

    recompute_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_badge_recompute"
    ]
    assert recompute_calls, "Expected run_badge_recompute call in badge audit page"

    for call in recompute_calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "include_non_live" not in kw_names

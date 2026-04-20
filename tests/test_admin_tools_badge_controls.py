from __future__ import annotations

import ast
from pathlib import Path


ADMIN_TOOLS = Path("jupr_app/ui/pages/admin_tools.py").read_text()
TREE = ast.parse(ADMIN_TOOLS)


def _call_nodes(func_name: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
            calls.append(node)
    return calls


def test_admin_tools_wires_badge_audit_call():
    calls = _call_nodes("build_badge_audit_report")
    assert calls, "Expected build_badge_audit_report call in admin_tools page"


def test_admin_tools_recompute_forces_scoped_strict_safety():
    calls = _call_nodes("run_badge_recompute")
    assert calls, "Expected run_badge_recompute call in admin_tools page"

    allow_strict_global_kwargs = [
        kw
        for call in calls
        for kw in call.keywords
        if kw.arg == "allow_strict_global"
    ]
    assert allow_strict_global_kwargs, "Expected allow_strict_global kwarg in run_badge_recompute call"
    assert any(isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in allow_strict_global_kwargs)


def test_admin_tools_has_strict_scope_guard_message():
    assert "Strict mode requires at least one scope filter" in ADMIN_TOOLS

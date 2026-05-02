import ast
import inspect
from pathlib import Path

from jupr_app.services.match_service import submit_match_batch

UI_PAGES = Path("jupr_app/ui/pages")


def test_ui_pages_do_not_import_process_matches_directly():
    offenders: list[str] = []
    for page in sorted(UI_PAGES.glob("*.py")):
        text = page.read_text(encoding="utf-8")
        if "from jupr_app.domain.match_processing import process_matches" in text:
            offenders.append(str(page))
        if "process_matches(" in text:
            offenders.append(str(page))
    assert not offenders, f"UI pages must use match service boundary, offenders: {offenders}"


def test_ui_submit_match_batch_kwargs_match_service_signature():
    accepted_kwargs = {
        name
        for name, parameter in inspect.signature(submit_match_batch).parameters.items()
        if parameter.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    accepted_kwargs.discard("ctx")
    accepted_kwargs.discard("matches")

    unexpected_by_file: dict[str, set[str]] = {}

    for page in sorted(UI_PAGES.glob("*.py")):
        tree = ast.parse(page.read_text(encoding="utf-8"), filename=str(page))
        used_kwargs: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "submit_match_batch":
                continue
            for keyword in node.keywords:
                if keyword.arg is not None:
                    used_kwargs.add(keyword.arg)

        unknown = used_kwargs - accepted_kwargs
        if unknown:
            unexpected_by_file[str(page)] = unknown

    assert not unexpected_by_file, (
        "UI submit_match_batch() keyword args must match service signature; "
        f"found mismatches: {unexpected_by_file}"
    )

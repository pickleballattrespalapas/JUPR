from pathlib import Path

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

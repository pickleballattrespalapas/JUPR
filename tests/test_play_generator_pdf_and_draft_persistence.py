from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_shared_pdf_helper_opens_a_new_tab_and_uses_session_storage() -> None:
    helper = read("lib/playGeneratorDraft.ts")
    assert 'window.open("", "_blank")' in helper
    assert "preparedWindow.location.replace(href)" in helper
    assert 'anchor.target = "_blank"' in helper
    assert "window.sessionStorage.getItem" in helper
    assert "window.sessionStorage.setItem" in helper
    assert "window.sessionStorage.removeItem" in helper


def test_public_and_admin_generators_preserve_unsaved_previews() -> None:
    for path in (
        "app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx",
        "app/admin/play-generators/GeneratorWorkspace.tsx",
    ):
        source = read(path)
        assert "readPlayGeneratorDraft<PreviewEvent>" in source
        assert "writePlayGeneratorDraft" in source
        assert "clearPlayGeneratorDraft(draftKey)" in source
        assert "Restored your unsaved schedule preview." in source
        assert "doc.save(" not in source
        assert "const pdfWindow = preparePdfWindow();" in source
        assert 'const blob = doc.output("blob")' in source
        assert "openPdfBlobInNewTab(blob, filename, pdfWindow)" in source
        assert "Download one-sheet PDF (opens new tab)" in source


def test_pdf_tab_is_prepared_before_async_jspdf_import() -> None:
    for path in (
        "app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx",
        "app/admin/play-generators/GeneratorWorkspace.tsx",
    ):
        source = read(path)
        prepared = source.index("const pdfWindow = preparePdfWindow();")
        imported = source.index('await import("jspdf")', prepared)
        assert prepared < imported

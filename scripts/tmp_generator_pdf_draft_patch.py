from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    (
        ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx",
        "public",
    ),
    (
        ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx",
        "admin",
    ),
)

IMPORT_MARKER = (
    'import GeneratorRosterSetup, { normalizeRosterName, recommendedGeneratorSetup, '
    'rosterNamesFromText } from "@/components/GeneratorRosterSetup";\n'
)
HELPER_IMPORT = '''import {
  clearPlayGeneratorDraft,
  closePreparedPdfWindow,
  openPdfBlobInNewTab,
  preparePdfWindow,
  readPlayGeneratorDraft,
  writePlayGeneratorDraft
} from "@/lib/playGeneratorDraft";
'''


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def function_block(text: str, name: str) -> tuple[int, int, str]:
    start = text.index(f"  async function {name}")
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                return start, end, text[start:end]
    raise RuntimeError(f"Could not find closing brace for {name}")


def patch_workspace(path: Path, scope: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        IMPORT_MARKER,
        IMPORT_MARKER + HELPER_IMPORT,
        label=f"{path}: helper import",
    )

    state_marker = '  const [message, setMessage] = useState<string | null>(null);\n'
    state_insert = state_marker + f'''  const [draftHydrated, setDraftHydrated] = useState(false);
  const draftKey = useMemo(
    () => `{scope}-play-generator-draft:${{clubId}}:${{generatorKind}}`,
    [clubId, generatorKind]
  );
'''
    text = replace_once(
        text,
        state_marker,
        state_insert,
        label=f"{path}: draft state",
    )

    sessions_effect = '''  useEffect(() => {
    void loadSessions();
'''
    persistence_effects = '''  useEffect(() => {
    setDraftHydrated(false);
    const stored = readPlayGeneratorDraft<PreviewEvent>(draftKey);
    if (stored) {
      setTitle(stored.title);
      setPlayFormat(stored.playFormat);
      setTargetCount(stored.targetCount);
      setParticipantText(stored.participantText);
      setLinkedPlayerIds(stored.linkedPlayerIds);
      setPreview(stored.preview);
      if (stored.preview) {
        setMessage("Restored your unsaved schedule preview.");
      }
    }
    setDraftHydrated(true);
  }, [draftKey]);

  useEffect(() => {
    if (!draftHydrated) return;
    writePlayGeneratorDraft(draftKey, {
      title,
      playFormat,
      targetCount,
      participantText,
      linkedPlayerIds,
      preview
    });
  }, [
    draftHydrated,
    draftKey,
    title,
    playFormat,
    targetCount,
    participantText,
    linkedPlayerIds,
    preview
  ]);

'''
    text = replace_once(
        text,
        sessions_effect,
        persistence_effects + sessions_effect,
        label=f"{path}: persistence effects",
    )

    start, end, start_block = function_block(text, "startSession")
    start_block = replace_once(
        start_block,
        "      const path = ",
        "      clearPlayGeneratorDraft(draftKey);\n      const path = ",
        label=f"{path}: clear draft on start",
    )
    text = text[:start] + start_block + text[end:]

    start, end, pdf_block = function_block(text, "downloadPdf")
    pdf_block = replace_once(
        pdf_block,
        '    const { jsPDF } = await import("jspdf");',
        '    const pdfWindow = preparePdfWindow();\n    try {\n      const { jsPDF } = await import("jspdf");',
        label=f"{path}: prepare PDF tab",
    )
    pdf_block = replace_once(
        pdf_block,
        '    doc.save(`${generatorSlug(generatorKind)}-schedule.pdf`);\n  }',
        '''    const filename = `${generatorSlug(generatorKind)}-schedule.pdf`;
    const blob = doc.output("blob");
    openPdfBlobInNewTab(blob, filename, pdfWindow);
    setMessage("Opened the PDF in a new tab. Your unsaved schedule remains here.");
    } catch (error) {
      closePreparedPdfWindow(pdfWindow);
      setMessage(error instanceof Error ? error.message : "Unable to open the schedule PDF.");
    }
  }''',
        label=f"{path}: PDF blob output",
    )
    text = text[:start] + pdf_block + text[end:]

    text = replace_once(
        text,
        "Download one-sheet PDF",
        "Open / print one-sheet PDF",
        label=f"{path}: PDF button label",
    )

    path.write_text(text, encoding="utf-8")


for file_path, storage_scope in FILES:
    patch_workspace(file_path, storage_scope)

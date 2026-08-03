export type StoredPlayGeneratorDraft<TPreview> = {
  version: 2;
  title: string;
  playFormat: "singles" | "doubles" | "doubles_singles";
  standingsSort?: "wins" | "points" | "differential";
  scoringMode?: "scored" | "unscored";
  targetCount: number;
  doublesCourtCount?: number;
  singlesCourtCount?: number;
  participantText: string;
  linkedPlayerIds: Record<string, number>;
  preview: TPreview | null;
  savedAt: string;
};

type DraftInput<TPreview> = Omit<StoredPlayGeneratorDraft<TPreview>, "version" | "savedAt">;

export function readPlayGeneratorDraft<TPreview>(
  storageKey: string
): StoredPlayGeneratorDraft<TPreview> | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredPlayGeneratorDraft<TPreview>> & { version?: number };
    if (
      ![1, 2].includes(Number(parsed.version)) ||
      typeof parsed.title !== "string" ||
      !["singles", "doubles", "doubles_singles"].includes(String(parsed.playFormat)) ||
      !Number.isFinite(Number(parsed.targetCount)) ||
      typeof parsed.participantText !== "string" ||
      !parsed.linkedPlayerIds ||
      typeof parsed.linkedPlayerIds !== "object"
    ) {
      window.sessionStorage.removeItem(storageKey);
      return null;
    }
    return {
      ...(parsed as StoredPlayGeneratorDraft<TPreview>),
      version: 2,
      doublesCourtCount: Number.isFinite(Number(parsed.doublesCourtCount))
        ? Math.max(0, Number(parsed.doublesCourtCount))
        : undefined,
      singlesCourtCount: Number.isFinite(Number(parsed.singlesCourtCount))
        ? Math.max(0, Number(parsed.singlesCourtCount))
        : undefined
    };
  } catch {
    return null;
  }
}

export function writePlayGeneratorDraft<TPreview>(
  storageKey: string,
  input: DraftInput<TPreview>
): void {
  if (typeof window === "undefined") return;
  try {
    const payload: StoredPlayGeneratorDraft<TPreview> = {
      ...input,
      version: 2,
      savedAt: new Date().toISOString()
    };
    window.sessionStorage.setItem(storageKey, JSON.stringify(payload));
  } catch {
    // Draft persistence is protective, not required for normal generator operation.
  }
}

export function clearPlayGeneratorDraft(storageKey: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(storageKey);
  } catch {
    // Ignore unavailable browser storage.
  }
}

export function preparePdfWindow(): Window | null {
  if (typeof window === "undefined") return null;
  const popup = window.open("", "_blank");
  if (!popup) return null;
  try {
    popup.opener = null;
    popup.document.open();
    popup.document.write(
      "<!doctype html><html><head><title>Preparing schedule PDF</title></head>" +
        "<body style='font-family:system-ui,sans-serif;padding:2rem'>Preparing schedule PDF…</body></html>"
    );
    popup.document.close();
  } catch {
    // The blank tab can still receive the generated blob URL.
  }
  return popup;
}

export function closePreparedPdfWindow(popup: Window | null): void {
  try {
    if (popup && !popup.closed) popup.close();
  } catch {
    // Ignore browser-specific popup restrictions.
  }
}

export function openPdfBlobInNewTab(
  blob: Blob,
  filename: string,
  preparedWindow: Window | null
): "new_tab" | "fallback" {
  const href = URL.createObjectURL(blob);
  const revokeLater = () => window.setTimeout(() => URL.revokeObjectURL(href), 120_000);

  if (preparedWindow && !preparedWindow.closed) {
    try {
      preparedWindow.location.replace(href);
      revokeLater();
      return "new_tab";
    } catch {
      // Continue to an explicit target=_blank fallback below.
    }
  }

  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.target = "_blank";
  anchor.rel = "noopener noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  revokeLater();
  return "fallback";
}

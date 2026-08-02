"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import GeneratorRosterSetup, { normalizeRosterName, recommendedGeneratorSetup, rosterNamesFromText } from "@/components/GeneratorRosterSetup";
import {
  clearPlayGeneratorDraft,
  closePreparedPdfWindow,
  openPdfBlobInNewTab,
  preparePdfWindow,
  readPlayGeneratorDraft,
  writePlayGeneratorDraft
} from "@/lib/playGeneratorDraft";
import { useAdminSession } from "@/lib/useAdminSession";

type GeneratorKind = "round_robin" | "ladder";
type PlayFormat = "singles" | "doubles";

type Participant = {
  id: string;
  name: string;
  player_id?: number | null;
  roster_order?: number;
};

type MatchRow = {
  id: string;
  round?: number;
  court?: number;
  miniRound?: number;
  sideA?: string[];
  sideB?: string[];
  teamA?: string[];
  teamB?: string[];
  scoreA?: number | null;
  scoreB?: number | null;
};

type RoundRow = {
  number: number;
  status: string;
  matches?: MatchRow[];
  courts?: Array<{
    courtNumber: number;
    participantIds?: string[];
    matches?: MatchRow[];
  }>;
  byeParticipantIds?: string[];
  warnings?: string[];
};

type PreviewEvent = {
  name: string;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  totalRounds: number;
  courtCount: number;
  previewFingerprint: string;
  participants: Participant[];
  rounds: RoundRow[];
};

type PreviewResponse = {
  ok: boolean;
  preview: PreviewEvent;
  schedule_rows: Array<Record<string, unknown>>;
};

type GeneratorSession = {
  session_key: string;
  title: string;
  status: string;
  version: string;
  generator_kind: GeneratorKind;
  play_format: PlayFormat;
  current_round_number?: number | null;
  total_rounds?: number | null;
  updated_at?: string | null;
};

type SessionListResponse = {
  ok: boolean;
  sessions: GeneratorSession[];
  count: number;
};

type StartResponse = {
  ok: boolean;
  session?: GeneratorSession;
};

type StatusResponse = {
  enabled: boolean;
  writes_enabled?: boolean;
  status: string;
  warnings?: string[];
};


type Props = {
  generatorKind: GeneratorKind;
  apiBase: string | null;
  clubId: string;
  status: StatusResponse | null;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const inputStyle = {
  width: "100%",
  boxSizing: "border-box" as const,
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const primaryButton = {
  border: 0,
  borderRadius: "999px",
  padding: "0.65rem 1rem",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  cursor: "pointer"
};

const secondaryButton = {
  border: "1px solid #cbd5e1",
  borderRadius: "999px",
  padding: "0.55rem 0.85rem",
  background: "white",
  color: "#0f172a",
  fontWeight: 800,
  cursor: "pointer"
};

function generatorSlug(kind: GeneratorKind): string {
  return kind === "round_robin" ? "round-robin-generator" : "ladder-generator";
}

function generatorTitle(kind: GeneratorKind): string {
  return kind === "round_robin" ? "Round-Robin Generator" : "Ladder Generator";
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function newKey(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `player-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function operationKey(): string {
  return `generator-${newKey()}`;
}

function flattenMatches(round: RoundRow): MatchRow[] {
  if (round.matches?.length) return round.matches;
  return (round.courts || []).flatMap((court) => court.matches || []);
}

function participantMap(event: PreviewEvent): Map<string, Participant> {
  return new Map((event.participants || []).map((participant) => [participant.id, participant]));
}

function sideLabel(ids: string[] | undefined, participants: Map<string, Participant>): string {
  return (ids || [])
    .map((id) => participants.get(String(id))?.name || String(id))
    .join(" / ");
}

function csvEscape(value: unknown): string {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadBlob(contents: BlobPart, type: string, filename: string): void {
  const blob = new Blob([contents], { type });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(href);
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

export default function GeneratorWorkspace({
  generatorKind,
  apiBase,
  clubId,
  status
}: Props) {
  const router = useRouter();
  const { accessToken } = useAdminSession();
  const [title, setTitle] = useState(
    `${generatorTitle(generatorKind)} ${new Date().toISOString().slice(0, 10)}`
  );
  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");
  const [targetCount, setTargetCount] = useState(8);
  const [participantText, setParticipantText] = useState("");
  const [linkedPlayerIds, setLinkedPlayerIds] = useState<Record<string, number>>({});
  const [preview, setPreview] = useState<PreviewEvent | null>(null);
  const [sessions, setSessions] = useState<GeneratorSession[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [draftHydrated, setDraftHydrated] = useState(false);
  const draftKey = useMemo(
    () => `admin-play-generator-draft:${clubId}:${generatorKind}`,
    [clubId, generatorKind]
  );
  const writesEnabled = status?.writes_enabled === true;

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const automaticSetup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount),
    [generatorKind, playFormat, targetCount]
  );
  const rosterReady = participantNames.length === targetCount;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing API base URL.");
    if (!accessToken) throw new Error("Sign in before using the generator.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), {
      ...options,
      headers,
      cache: "no-store"
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(String(payload?.detail || `API error (${response.status})`));
    }
    return payload as T;
  }

  async function loadSessions(): Promise<void> {
    if (!accessToken || !status?.enabled) return;
    try {
      const payload = await requestJson<SessionListResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions?generator_kind=${generatorKind}&limit=100`
      );
      setSessions(payload.sessions || []);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load sessions.");
    }
  }

  useEffect(() => {
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

  useEffect(() => {
    void loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, generatorKind]);

  function invalidatePreview(): void {
    setPreview(null);
    setMessage(null);
  }

  function requestBody(): Record<string, unknown> {
    if (!rosterReady) {
      throw new Error(`Add exactly ${targetCount} unique players before previewing.`);
    }
    const orderedIds = participantNames.map((name) =>
      Number(linkedPlayerIds[normalizeRosterName(name)] || 0)
    );
    const allLinked = orderedIds.every((playerId) => playerId > 0);
    return {
      generator_kind: generatorKind,
      play_format: playFormat,
      title: title.trim(),
      participant_names: participantNames,
      player_ids: allLinked ? orderedIds : [],
      total_rounds: automaticSetup.totalRounds,
      court_count: automaticSetup.courtCount
    };
  }

  async function generatePreview(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      const body = requestBody();
      const payload = await requestJson<PreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/preview`,
        { method: "POST", body: JSON.stringify(body) }
      );
      setPreview(payload.preview);
      setMessage(
        generatorKind === "round_robin"
          ? `Previewed ${payload.preview.rounds.length} planned round(s).`
          : "Previewed Round 1. Later ladder rounds are generated from results."
      );
    } catch (error) {
      setPreview(null);
      setMessage(error instanceof Error ? error.message : "Unable to preview the schedule.");
    } finally {
      setBusy(false);
    }
  }

  async function startSession(): Promise<void> {
    if (!preview) return;
    setBusy(true);
    setMessage(null);
    try {
      const body = {
        ...requestBody(),
        preview_fingerprint: preview.previewFingerprint,
        expected_version: "new",
        idempotency_key: operationKey()
      };
      const payload = await requestJson<StartResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions`,
        { method: "POST", body: JSON.stringify(body) }
      );
      const session = payload.session;
      if (!session?.session_key) throw new Error("The session was created without a session key.");
      clearPlayGeneratorDraft(draftKey);
      const path = `/admin/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
        session.session_key
      )}/rounds/${session.current_round_number || 1}`;
      router.push(path);
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to start the session.");
    } finally {
      setBusy(false);
    }
  }

  function downloadCsv(): void {
    if (!preview) return;
    const participants = participantMap(preview);
    const rows = [
      ["Round", "Court", "Mini round", "Side A", "Score A", "Score B", "Side B", "Byes"]
    ];
    for (const round of preview.rounds) {
      const byes = (round.byeParticipantIds || [])
        .map((id) => participants.get(id)?.name || id)
        .join(" / ");
      const matches = flattenMatches(round);
      if (!matches.length) {
        rows.push([String(round.number), "", "", "", "", "", "", byes]);
      }
      for (const match of matches) {
        rows.push([
          String(round.number),
          String(match.court || ""),
          String(match.miniRound || ""),
          sideLabel(match.sideA || match.teamA, participants),
          "",
          "",
          sideLabel(match.sideB || match.teamB, participants),
          byes
        ]);
      }
    }
    const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\n");
    downloadBlob(csv, "text/csv;charset=utf-8", `${generatorSlug(generatorKind)}-schedule.csv`);
  }

  async function downloadPdf(): Promise<void> {
    if (!preview) return;
    const pdfWindow = preparePdfWindow();
    try {
      const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ orientation: "landscape", unit: "pt", format: "letter" });
    const participants = participantMap(preview);
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 24;
    const titleHeight = 34;
    const blocks = preview.rounds.map((round) => {
      const lines = [`Round ${round.number}`];
      for (const match of flattenMatches(round)) {
        const court = match.court ? `C${match.court}` : "";
        const mini = match.miniRound ? `.${match.miniRound}` : "";
        lines.push(
          `${court}${mini} ${sideLabel(match.sideA || match.teamA, participants)}  ___ - ___  ${sideLabel(
            match.sideB || match.teamB,
            participants
          )}`.trim()
        );
      }
      const byeNames = (round.byeParticipantIds || [])
        .map((id) => participants.get(id)?.name || id)
        .join(", ");
      if (byeNames) lines.push(`Byes: ${byeNames}`);
      return lines;
    });
    const totalLines = blocks.reduce((sum, lines) => sum + lines.length + 1, 0);
    const columns = totalLines > 90 ? 4 : totalLines > 56 ? 3 : 2;
    const availableHeight = pageHeight - margin * 2 - titleHeight;
    const linesPerColumn = Math.max(1, Math.ceil(totalLines / columns));
    const fontSize = Math.max(4.5, Math.min(8.5, availableHeight / (linesPerColumn * 1.28)));
    const lineHeight = fontSize * 1.28;
    const columnWidth = (pageWidth - margin * 2) / columns;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text(`${preview.name} - ${playFormat === "singles" ? "Singles" : "Doubles"}`, margin, margin + 11);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(7);
    doc.text(
      generatorKind === "ladder"
        ? "Ladder preview: Round 1 only. Later rounds depend on results."
        : `${preview.totalRounds} planned rounds`,
      margin,
      margin + 24
    );

    let column = 0;
    let y = margin + titleHeight;
    for (const lines of blocks) {
      const blockHeight = (lines.length + 1) * lineHeight;
      if (y + blockHeight > pageHeight - margin && column < columns - 1) {
        column += 1;
        y = margin + titleHeight;
      }
      const x = margin + column * columnWidth;
      lines.forEach((line, index) => {
        doc.setFont("helvetica", index === 0 ? "bold" : "normal");
        doc.setFontSize(index === 0 ? fontSize + 0.5 : fontSize);
        const clipped = doc.splitTextToSize(line, columnWidth - 10)[0] || line;
        doc.text(clipped, x, y);
        y += lineHeight;
      });
      y += lineHeight * 0.45;
    }
    const filename = `${generatorSlug(generatorKind)}-schedule.pdf`;
    const blob = doc.output("blob");
    openPdfBlobInNewTab(blob, filename, pdfWindow);
    setMessage("Opened the PDF in a new tab. Your unsaved schedule remains here.");
    } catch (error) {
      closePreparedPdfWindow(pdfWindow);
      setMessage(error instanceof Error ? error.message : "Unable to open the schedule PDF.");
    }
  }

  const previewParticipants = preview ? participantMap(preview) : new Map<string, Participant>();

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {!status?.enabled ? (
        <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>
          <strong>{generatorTitle(generatorKind)} is disabled.</strong>
          <p>{status?.warnings?.[0] || "Enable the generator backend before testing."}</p>
        </article>
      ) : null}

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Set up the session</h2>
        <p style={{ color: "#475569" }}>
          Enter players in the starting order you want. Reorder them before previewing to change
          the schedule and bye order.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: "0.75rem",
            marginBottom: "1rem"
          }}
        >
          <label>
            Session title
            <br />
            <input
              value={title}
              onChange={(event) => {
                setTitle(event.target.value);
                invalidatePreview();
              }}
              style={inputStyle}
            />
          </label>
          <label>
            Play format
            <br />
            <select
              value={playFormat}
              onChange={(event) => {
                const nextFormat = event.target.value as PlayFormat;
                setPlayFormat(nextFormat);
                setTargetCount((current) => Math.max(current, nextFormat === "singles" ? 2 : 4));
                invalidatePreview();
              }}
              style={inputStyle}
            >
              <option value="doubles">Doubles</option>
              <option value="singles">Singles</option>
            </select>
          </label>
        </div>

        <GeneratorRosterSetup
          apiBase={apiBase}
          clubKey={clubId}
          generatorKind={generatorKind}
          playFormat={playFormat}
          targetCount={targetCount}
          participantText={participantText}
          linkedPlayerIds={linkedPlayerIds}
          onTargetCountChange={setTargetCount}
          onParticipantTextChange={setParticipantText}
          onLinkedPlayerIdsChange={setLinkedPlayerIds}
          onInvalidate={invalidatePreview}
        />
        <div style={{ marginTop: "0.9rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={() => void generatePreview()}
            disabled={busy || !rosterReady}
            style={primaryButton}
          >
            {busy ? "Generating…" : "Preview matchups"}
          </button>
        </div>
        {message ? (
          <p
            role="status"
            aria-live="polite"
            style={{ color: /unable|error|must|requires|changed/i.test(message) ? "#b91c1c" : "#166534" }}
          >
            {message}
          </p>
        ) : null}
      </article>

      {preview ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>2. Preview before starting</h2>
          <p style={{ color: "#475569" }}>
            {generatorKind === "ladder"
              ? "Only Round 1 is shown. Round 2 and later are generated from saved results."
              : "Review every planned round, matchup, and bye. Change the roster order above and regenerate when needed."}
          </p>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            <button type="button" onClick={downloadCsv} style={secondaryButton}>
              Download CSV
            </button>
            <button type="button" onClick={() => void downloadPdf()} style={secondaryButton}>
              Download one-sheet PDF (opens new tab)
            </button>
            <button
              type="button"
              onClick={() => void startSession()}
              disabled={busy || !writesEnabled}
              style={primaryButton}
            >
              {busy ? "Starting…" : "Start session"}
            </button>
          </div>

          <div style={{ display: "grid", gap: "0.85rem" }}>
            {preview.rounds.map((round) => {
              const byes = (round.byeParticipantIds || [])
                .map((id) => previewParticipants.get(id)?.name || id)
                .join(", ");
              return (
                <section
                  key={round.number}
                  style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem" }}
                >
                  <h3 style={{ marginTop: 0 }}>Round {round.number}</h3>
                  <div style={{ display: "grid", gap: "0.45rem" }}>
                    {flattenMatches(round).map((match) => (
                      <div
                        key={match.id}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "4.5rem minmax(0, 1fr) auto minmax(0, 1fr)",
                          gap: "0.6rem",
                          alignItems: "center",
                          padding: "0.45rem",
                          background: "#f8fafc",
                          borderRadius: "8px"
                        }}
                      >
                        <span style={{ color: "#64748b" }}>
                          Court {match.court || "—"}
                          {match.miniRound ? `.${match.miniRound}` : ""}
                        </span>
                        <strong>{sideLabel(match.sideA || match.teamA, previewParticipants)}</strong>
                        <span>vs.</span>
                        <strong style={{ textAlign: "right" }}>
                          {sideLabel(match.sideB || match.teamB, previewParticipants)}
                        </strong>
                      </div>
                    ))}
                  </div>
                  {byes ? <p style={{ marginBottom: 0 }}><strong>Byes:</strong> {byes}</p> : null}
                  {(round.warnings || []).map((warning) => (
                    <p key={warning} style={{ marginBottom: 0, color: "#92400e" }}>
                      {warning}
                    </p>
                  ))}
                </section>
              );
            })}
          </div>
        </article>
      ) : null}

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>Existing {generatorTitle(generatorKind)} sessions</h2>
            <p style={{ color: "#475569" }}>Resume an active round or review a completed session.</p>
          </div>
          <button type="button" onClick={() => void loadSessions()} disabled={busy} style={secondaryButton}>
            Refresh
          </button>
        </div>
        {sessions.length ? (
          <div style={{ display: "grid", gap: "0.65rem" }}>
            {sessions.map((session) => (
              <article
                key={session.session_key}
                style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
                  <div>
                    <strong>{session.title}</strong>
                    <p style={{ margin: "0.25rem 0 0", color: "#475569" }}>
                      {session.play_format} · Round {session.current_round_number || 1} of {session.total_rounds || "?"}
                      {" · "}{session.status}
                    </p>
                  </div>
                  <Link
                    href={`/admin/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
                      session.session_key
                    )}/rounds/${session.current_round_number || 1}`}
                    style={{ fontWeight: 800 }}
                  >
                    Open session
                  </Link>
                </div>
                <p style={{ marginBottom: 0, color: "#64748b", fontSize: "0.85rem" }}>
                  Updated {formatTimestamp(session.updated_at)}
                </p>
              </article>
            ))}
          </div>
        ) : (
          <p style={{ color: "#64748b" }}>No generator sessions yet.</p>
        )}
      </article>
    </div>
  );
}

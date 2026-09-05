"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import GeneratorRosterSetup, { normalizeRosterName, recommendedGeneratorSetup, recommendedMixedCourtSetup, rosterNamesFromText } from "@/components/GeneratorRosterSetup";
import {
  clearPlayGeneratorDraft,
  closePreparedPdfWindow,
  openPdfBlobInNewTab,
  preparePdfWindow,
  readPlayGeneratorDraft,
  writePlayGeneratorDraft
} from "@/lib/playGeneratorDraft";

type GeneratorKind = "round_robin" | "ladder";
type PlayFormat = "singles" | "doubles" | "doubles_singles";
type StandingsSort = "wins" | "points" | "differential";
type ScoringMode = "scored" | "unscored";

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
  playFormat?: "singles" | "doubles";
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
  formatCounts?: { doubles?: number; singles?: number };
};

type PreviewEvent = {
  name: string;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  standingsSort?: StandingsSort;
  scoringMode?: ScoringMode;
  totalRounds: number;
  courtCount: number;
  doublesCourtCount: number;
  singlesCourtCount: number;
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
  scoring_mode?: ScoringMode;
  current_round_number?: number | null;
  total_rounds?: number | null;
  doubles_court_count?: number;
  singles_court_count?: number;
  updated_at?: string | null;
};

type SessionListResponse = {
  ok: boolean;
  sessions: GeneratorSession[];
  count: number;
};

type StartResponse = {
  ok: boolean;
  edit_token?: string;
  session?: GeneratorSession;
};

type PendingStartPayload = Record<string, unknown> & {
  idempotency_key: string;
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

function playFormatLabel(playFormat: PlayFormat | string): string {
  if (playFormat === "singles") return "Singles";
  if (playFormat === "doubles_singles") return "Doubles + Singles Mix";
  return "Doubles";
}

function matchFormatLabel(match: MatchRow, eventFormat: PlayFormat): string {
  if (match.playFormat === "singles" || match.playFormat === "doubles") {
    return playFormatLabel(match.playFormat);
  }
  const sideSize = (match.sideA || match.teamA || []).length;
  return playFormatLabel(sideSize === 1 ? "singles" : eventFormat === "doubles_singles" ? "doubles" : eventFormat);
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

class ApiRequestError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
  }
}

class GeneratorSetupError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GeneratorSetupError";
  }
}

function workspaceErrorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiRequestError || error instanceof GeneratorSetupError
    ? error.message
    : fallback;
}

function readPendingStartPayload(storageKey: string): PendingStartPayload | null {
  let raw: string | null = null;
  try {
    raw = sessionStorage.getItem(storageKey);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const payload = JSON.parse(raw) as Record<string, unknown> | null;
    if (
      payload &&
      typeof payload === "object" &&
      typeof payload.idempotency_key === "string" &&
      payload.idempotency_key
    ) {
      return payload as PendingStartPayload;
    }
  } catch {
    // A partial or old browser record cannot be retried safely.
  }
  clearPendingStartPayload(storageKey);
  return null;
}

function clearPendingStartPayload(storageKey: string): boolean {
  try {
    sessionStorage.removeItem(storageKey);
    return sessionStorage.getItem(storageKey) === null;
  } catch {
    return false;
  }
}

function isDefinitiveStartRejection(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && [400, 403, 409, 422].includes(error.status);
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

function sessionStatusLabel(status: string): string {
  if (status === "completed") return "Complete";
  if (status === "active") return "In progress";
  return "Ready";
}

export default function GeneratorWorkspace({
  generatorKind,
  apiBase,
  clubId,
  status
}: Props) {
  const router = useRouter();
  const [title, setTitle] = useState(
    `${generatorTitle(generatorKind)} ${new Date().toISOString().slice(0, 10)}`
  );
  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");
  const [standingsSort, setStandingsSort] = useState<StandingsSort>("wins");
  const [scoringMode, setScoringMode] = useState<ScoringMode>("scored");
  const [targetCount, setTargetCount] = useState(8);
  const initialMixedSetup = recommendedMixedCourtSetup(8);
  const [doublesCourtCount, setDoublesCourtCount] = useState(initialMixedSetup.doublesCourtCount);
  const [singlesCourtCount, setSinglesCourtCount] = useState(initialMixedSetup.singlesCourtCount);
  const [participantText, setParticipantText] = useState("");
  const [linkedPlayerIds, setLinkedPlayerIds] = useState<Record<string, number>>({});
  const [preview, setPreview] = useState<PreviewEvent | null>(null);
  const [sessions, setSessions] = useState<GeneratorSession[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [draftHydrated, setDraftHydrated] = useState(false);
  const [pendingStart, setPendingStart] = useState(false);
  const [startSucceeded, setStartSucceeded] = useState(false);
  const draftKey = useMemo(
    () => `public-play-generator-draft:${clubId}:${generatorKind}`,
    [clubId, generatorKind]
  );
  const startOperationStorageKey = useMemo(
    () => `public-play-generator-create:${clubId}:${generatorKind}`,
    [clubId, generatorKind]
  );
  const writesEnabled = status?.writes_enabled === true;

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const automaticSetup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount),
    [generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount]
  );
  const rosterReady = participantNames.length === targetCount;
  const mixedSetupValid = playFormat !== "doubles_singles" || (
    doublesCourtCount >= 1 &&
    singlesCourtCount >= 1 &&
    doublesCourtCount * 4 + singlesCourtCount * 2 <= targetCount
  );

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) {
      throw new ApiRequestError("This play tool is temporarily unavailable. Please try again.", 503);
    }
    const headers = new Headers(options?.headers);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), {
      ...options,
      headers,
      cache: "no-store"
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      if (
        payload?.detail === "Score entry is temporarily unavailable." ||
        payload?.detail === "Live sessions are temporarily unavailable. Please try again later."
      ) {
        throw new ApiRequestError("This play tool is temporarily unavailable.", response.status);
      }
      throw new ApiRequestError("We couldn't complete that request. Please try again.", response.status);
    }
    return payload as T;
  }

  async function loadSessions(): Promise<void> {
    if (!status?.enabled) return;
    try {
      const payload = await requestJson<SessionListResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions?generator_kind=${generatorKind}&limit=100`
      );
      setSessions(payload.sessions || []);
    } catch {
      setMessage("We couldn't load saved sessions. Please try again.");
    }
  }

  useEffect(() => {
    setStartSucceeded(false);
    setDraftHydrated(false);
    const stored = readPlayGeneratorDraft<PreviewEvent>(draftKey);
    if (stored) {
      setTitle(stored.title);
      setPlayFormat(stored.playFormat);
      setStandingsSort(stored.standingsSort || "wins");
      setScoringMode(generatorKind === "round_robin" ? stored.scoringMode || "scored" : "scored");
      setTargetCount(stored.targetCount);
      const mixedSetup = recommendedMixedCourtSetup(stored.targetCount);
      setDoublesCourtCount(stored.doublesCourtCount || mixedSetup.doublesCourtCount);
      setSinglesCourtCount(stored.singlesCourtCount || mixedSetup.singlesCourtCount);
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
    setPendingStart(Boolean(readPendingStartPayload(startOperationStorageKey)));
  }, [startOperationStorageKey]);

  useEffect(() => {
    if (!draftHydrated) return;
    writePlayGeneratorDraft(draftKey, {
      title,
      playFormat,
      standingsSort,
      scoringMode,
      targetCount,
      doublesCourtCount,
      singlesCourtCount,
      participantText,
      linkedPlayerIds,
      preview
    });
  }, [
    draftHydrated,
    draftKey,
    title,
    playFormat,
    standingsSort,
    scoringMode,
    targetCount,
    doublesCourtCount,
    singlesCourtCount,
    participantText,
    linkedPlayerIds,
    preview
  ]);

  useEffect(() => {
    void loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generatorKind, status?.enabled]);

  const invalidatePreview = useCallback((): void => {
    setPreview(null);
    setMessage(null);
  }, []);

  function requestBody(): Record<string, unknown> {
    if (!rosterReady) {
      throw new GeneratorSetupError(`Add exactly ${targetCount} unique players before previewing.`);
    }
    if (!mixedSetupValid) {
      throw new GeneratorSetupError("Choose a doubles and singles court mix that fits the selected player count.");
    }
    const participantPlayerIds = Object.fromEntries(
      participantNames.flatMap((name) => {
        const playerId = Number(linkedPlayerIds[normalizeRosterName(name)] || 0);
        return playerId > 0 ? [[name, playerId]] : [];
      })
    );
    return {
      generator_kind: generatorKind,
      play_format: playFormat,
      standings_sort: standingsSort,
      scoring_mode: generatorKind === "round_robin" ? scoringMode : "scored",
      title: title.trim(),
      participant_names: participantNames,
      participant_player_ids: participantPlayerIds,
      total_rounds: automaticSetup.totalRounds,
      court_count: automaticSetup.courtCount,
      doubles_court_count: automaticSetup.doublesCourtCount,
      singles_court_count: automaticSetup.singlesCourtCount
    };
  }

  async function generatePreview(): Promise<void> {
    setBusy(true);
    setMessage(null);
    try {
      const body = requestBody();
      const payload = await requestJson<PreviewResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/preview`,
        { method: "POST", body: JSON.stringify(body) }
      );
      setPreview(payload.preview);
      setMessage(
        generatorKind === "round_robin"
          ? `Preview ready: ${payload.preview.rounds.length} round${payload.preview.rounds.length === 1 ? "" : "s"}.`
          : "Round 1 is ready. Each later round will be based on the saved scores."
      );
    } catch (error) {
      setPreview(null);
      setMessage(workspaceErrorMessage(error, "We couldn't preview the schedule. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  async function startSession(): Promise<void> {
    if (startSucceeded) return;
    let requestPayload = readPendingStartPayload(startOperationStorageKey);
    const isRecoveryAttempt = Boolean(requestPayload);
    if (!requestPayload && !preview) return;
    if (!apiBase) {
      setPendingStart(Boolean(requestPayload));
      setMessage("This play tool is temporarily unavailable. Please try again later.");
      return;
    }

    if (!requestPayload && preview) {
      try {
        requestPayload = {
          ...requestBody(),
          preview_fingerprint: preview.previewFingerprint,
          idempotency_key: operationKey()
        };
      } catch (error) {
        setMessage(
          workspaceErrorMessage(
            error,
            "We couldn't prepare the session. Please check the setup and try again."
          )
        );
        return;
      }
      try {
        sessionStorage.setItem(startOperationStorageKey, JSON.stringify(requestPayload));
      } catch {
        setMessage("Your browser couldn't save this start attempt, so no session was started. Please try again.");
        return;
      }
    }
    if (!requestPayload) return;

    setPendingStart(isRecoveryAttempt);
    setBusy(true);
    setMessage(null);
    let sessionConfirmed = false;
    try {
      const payload = await requestJson<StartResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions`,
        { method: "POST", body: JSON.stringify(requestPayload) }
      );
      const session = payload.session;
      const editToken = String(payload.edit_token || "");
      if (!session?.session_key || !editToken) {
        throw new Error("The session start response was incomplete.");
      }
      sessionStorage.setItem(`public-generator-edit:${clubId}:${session.session_key}`, editToken);
      sessionConfirmed = true;
      setStartSucceeded(true);
      setPendingStart(!clearPendingStartPayload(startOperationStorageKey));
      clearPlayGeneratorDraft(draftKey);
      const path = `/clubs/${clubId}/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
        session.session_key
      )}/rounds/${session.current_round_number || 1}`;
      router.push(`${path}#edit=${encodeURIComponent(editToken)}`);
      router.refresh();
    } catch (error) {
      if (sessionConfirmed) {
        setStartSucceeded(true);
        setPendingStart(!clearPendingStartPayload(startOperationStorageKey));
        setMessage("The session started, but it didn't open automatically. Open it from Recent sessions.");
      } else if (isDefinitiveStartRejection(error)) {
        setPendingStart(!clearPendingStartPayload(startOperationStorageKey));
        setMessage(error.message);
      } else {
        setPendingStart(true);
        setMessage(
          "We couldn't confirm that the session started. Try again here before creating another one; we'll resend the same setup."
        );
      }
    } finally {
      setBusy(false);
    }
  }

  function downloadCsv(): void {
    if (!preview) return;
    const participants = participantMap(preview);
    const rows = [
      ["Round", "Format", "Court", "Mini round", "Side A", "Score A", "Score B", "Side B", "Byes"]
    ];
    for (const round of preview.rounds) {
      const byes = (round.byeParticipantIds || [])
        .map((id) => participants.get(id)?.name || id)
        .join(" / ");
      const matches = flattenMatches(round);
      if (!matches.length) {
        rows.push([String(round.number), "", "", "", "", "", "", "", byes]);
      }
      for (const match of matches) {
        rows.push([
          String(round.number),
          matchFormatLabel(match, preview.playFormat),
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
          `${matchFormatLabel(match, preview.playFormat)} ${court}${mini} ${sideLabel(match.sideA || match.teamA, participants)}  ___ - ___  ${sideLabel(
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
    doc.text(`${preview.name} - ${playFormatLabel(preview.playFormat)}`, margin, margin + 11);
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
    } catch {
      closePreparedPdfWindow(pdfWindow);
      setMessage("We couldn't open the schedule PDF. Please try again.");
    }
  }

  const previewParticipants = preview ? participantMap(preview) : new Map<string, Participant>();

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {!status?.enabled ? (
        <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>
          <strong>This play tool is temporarily unavailable.</strong>
          <p>Please try again later.</p>
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
                const nextCount = Math.max(targetCount, nextFormat === "singles" ? 2 : nextFormat === "doubles_singles" ? 6 : 4);
                setPlayFormat(nextFormat);
                setTargetCount(nextCount);
                if (nextFormat === "doubles_singles") {
                  const mixedSetup = recommendedMixedCourtSetup(nextCount);
                  setDoublesCourtCount(mixedSetup.doublesCourtCount);
                  setSinglesCourtCount(mixedSetup.singlesCourtCount);
                }
                invalidatePreview();
              }}
              style={inputStyle}
            >
              <option value="doubles">Doubles</option>
              <option value="singles">Singles</option>
              {generatorKind === "round_robin" ? (
                <option value="doubles_singles">Doubles + Singles Mix</option>
              ) : null}
            </select>
          </label>
          {generatorKind === "round_robin" ? (
            <>
              <label>
                Round scoring
                <br />
                <select
                  value={scoringMode}
                  onChange={(event) => {
                    setScoringMode(event.target.value as ScoringMode);
                    invalidatePreview();
                  }}
                  style={inputStyle}
                >
                  <option value="scored">Scored — enter scores and show standings</option>
                  <option value="unscored">Unscored — mark each round played</option>
                </select>
                <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>
                  Unscored sessions have no score fields or standings between rounds.
                </small>
              </label>
              {scoringMode === "scored" ? (
                <label>
                  Standings ranked by
                  <br />
                  <select
                    value={standingsSort}
                    onChange={(event) => {
                      setStandingsSort(event.target.value as StandingsSort);
                      invalidatePreview();
                    }}
                    style={inputStyle}
                  >
                    <option value="wins">Total wins</option>
                    <option value="points">Total points</option>
                    <option value="differential">Point differential</option>
                  </select>
                  <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>
                    Standings will be ranked by this stat.
                  </small>
                </label>
              ) : null}
            </>
          ) : null}
        </div>

        <GeneratorRosterSetup
          apiBase={apiBase}
          clubKey={clubId}
          generatorKind={generatorKind}
          playFormat={playFormat}
          targetCount={targetCount}
          participantText={participantText}
          linkedPlayerIds={linkedPlayerIds}
          doublesCourtCount={doublesCourtCount}
          singlesCourtCount={singlesCourtCount}
          onTargetCountChange={setTargetCount}
          onDoublesCourtCountChange={setDoublesCourtCount}
          onSinglesCourtCountChange={setSinglesCourtCount}
          onParticipantTextChange={setParticipantText}
          onLinkedPlayerIdsChange={setLinkedPlayerIds}
          onInvalidate={invalidatePreview}
        />
        <div style={{ marginTop: "0.9rem", display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={() => void generatePreview()}
            disabled={busy || pendingStart || startSucceeded || !rosterReady || !mixedSetupValid}
            style={primaryButton}
          >
            {busy ? "Generating…" : "Preview matchups"}
          </button>
        </div>
        {message ? (
          <p
            role="status"
            aria-live="polite"
            style={{ color: /couldn|unable|error|must|requires|changed/i.test(message) ? "#b91c1c" : "#166534" }}
          >
            {message}
          </p>
        ) : null}
      </article>

      {pendingStart && !startSucceeded ? (
        <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>
          <h2 style={{ marginTop: 0 }}>Session start not confirmed</h2>
          <p>
            Try again here before starting another session. We’ll resend the same setup so you
            don’t create a duplicate.
          </p>
          <button
            type="button"
            onClick={() => void startSession()}
            disabled={busy || !writesEnabled}
            style={primaryButton}
          >
            {busy ? "Trying again…" : "Try starting again"}
          </button>
        </article>
      ) : null}

      {preview ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>2. Preview before starting</h2>
          <p style={{ color: "#475569" }}>
            {generatorKind === "ladder"
              ? "Only Round 1 is shown. Round 2 and later are generated from saved results."
              : scoringMode === "unscored"
                ? "Review every planned round, matchup, and bye. During play, use Mark round played to move directly to the next round."
                : "Review every planned round, matchup, and bye. Change the roster order above and regenerate when needed."}
          </p>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            <button type="button" onClick={downloadCsv} style={secondaryButton}>
              Download CSV
            </button>
            <button type="button" onClick={() => void downloadPdf()} style={secondaryButton}>
              Download one-sheet PDF (opens new tab)
            </button>
            {!pendingStart && !startSucceeded ? (
              <button
                type="button"
                onClick={() => void startSession()}
                disabled={busy || !writesEnabled}
                style={primaryButton}
              >
                {busy ? "Starting…" : "Start session"}
              </button>
            ) : null}
          </div>

          <div style={{ display: "grid", gap: "0.85rem" }}>
            {preview.rounds.map((round) => {
              const roundMatches = flattenMatches(round);
              const byes = (round.byeParticipantIds || [])
                .map((id) => previewParticipants.get(id)?.name || id)
                .join(", ");
              const doublesGames = round.formatCounts?.doubles ?? roundMatches.filter(
                (match) => matchFormatLabel(match, preview.playFormat) === "Doubles"
              ).length;
              const singlesGames = round.formatCounts?.singles ?? roundMatches.filter(
                (match) => matchFormatLabel(match, preview.playFormat) === "Singles"
              ).length;
              return (
                <section
                  key={round.number}
                  style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.85rem" }}
                >
                  <h3 style={{ marginTop: 0 }}>Round {round.number}</h3>
                  {preview.playFormat === "doubles_singles" ? (
                    <p style={{ margin: "-0.25rem 0 0.65rem", color: "#475569" }}>
                      {doublesGames} doubles game{doublesGames === 1 ? "" : "s"} · {singlesGames} singles game{singlesGames === 1 ? "" : "s"}
                    </p>
                  ) : null}
                  <div style={{ display: "grid", gap: "0.45rem" }}>
                    {roundMatches.map((match) => (
                      <div
                        key={match.id}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "7.5rem minmax(0, 1fr) auto minmax(0, 1fr)",
                          gap: "0.6rem",
                          alignItems: "center",
                          padding: "0.45rem",
                          background: "#f8fafc",
                          borderRadius: "8px"
                        }}
                      >
                        <span style={{ color: "#64748b" }}>
                          {matchFormatLabel(match, preview.playFormat)} · Court {match.court || "—"}
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
            <h2 style={{ marginTop: 0 }}>Recent sessions</h2>
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
                      {playFormatLabel(session.play_format)} · Round {session.current_round_number || 1} of {session.total_rounds || "?"}
                      {" · "}{sessionStatusLabel(session.status)}
                    </p>
                  </div>
                  <Link
                    href={`/clubs/${clubId}/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
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
          <p style={{ color: "#64748b" }}>No saved sessions yet.</p>
        )}
      </article>
    </div>
  );
}

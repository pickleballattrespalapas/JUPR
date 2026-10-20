"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import GeneratorRosterSetup, { normalizeRosterName, recommendedGeneratorSetup, recommendedMixedCourtSetup, rosterNamesFromText } from "@/components/GeneratorRosterSetup";
import GeneratorMoneyballExpectations, { type MoneyballExpectation } from "@/components/GeneratorMoneyballExpectations";
import {
  clearPlayGeneratorDraft,
  closePreparedPdfWindow,
  openPdfBlobInNewTab,
  preparePdfWindow,
  readPlayGeneratorDraft,
  writePlayGeneratorDraft
} from "@/lib/playGeneratorDraft";
import { useAdminSession } from "@/lib/useAdminSession";
import { GeneratorRequestError, generatorStartFailure, isRejectedGeneratorStart, readGeneratorStart, writeGeneratorStart, type CreatedGeneratorStart, type GeneratorStartRecovery, type PendingGeneratorStart } from "@/lib/adminGeneratorStart";

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
  courtLabel?: string;
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
  ratingMode?: "rated" | "unrated";
  totalRounds: number;
  courtCount: number;
  doublesCourtCount: number;
  singlesCourtCount: number;
  previewFingerprint: string;
  participants: Participant[];
  rounds: RoundRow[];
  moneyballExpectations?: MoneyballExpectation[];
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
  round_robin_hub_id?: string | null;
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
  embedded?: boolean;
  onCreated?: (sessionKey: string) => void;
  returnHubId?: string;
  draftToken?: string;
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
  status,
  embedded = false,
  onCreated,
  returnHubId,
  draftToken
}: Props) {
  const router = useRouter();
  const { accessToken, session: adminSession } = useAdminSession();
  const creationClubId = clubId;
  const canCreatePlayer = adminSession?.capabilities?.assignments.some(assignment => assignment.club_id === creationClubId && assignment.permissions.includes("manage_players"));
  const [creationPending, setCreationPending] = useState(false);
  const [title, setTitle] = useState(
    `${generatorTitle(generatorKind)} ${new Date().toISOString().slice(0, 10)}`
  );
  const [eventFormat, setEventFormat] = useState<"standard" | "moneyball_expectations">("standard");
  const moneyball = generatorKind === "round_robin" && eventFormat === "moneyball_expectations";
  const [playFormat, setPlayFormat] = useState<PlayFormat>("doubles");
  const [standingsSort, setStandingsSort] = useState<StandingsSort>("wins");
  const [scoringMode, setScoringMode] = useState<ScoringMode>("scored");
  const [ratingMode, setRatingMode] = useState<"rated" | "unrated">("unrated");
  const [targetCount, setTargetCount] = useState(8);
  const initialMixedSetup = recommendedMixedCourtSetup(8);
  const [doublesCourtCount, setDoublesCourtCount] = useState(initialMixedSetup.doublesCourtCount);
  const [singlesCourtCount, setSinglesCourtCount] = useState(initialMixedSetup.singlesCourtCount);
  const [participantText, setParticipantText] = useState("");
  const [linkedPlayerIds, setLinkedPlayerIds] = useState<Record<string, number>>({});
  const [preview, setPreview] = useState<PreviewEvent | null>(null);
  const [sessions, setSessions] = useState<GeneratorSession[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ text: string; kind: "info" | "success" | "error" } | null>(null);
  const [hydratedDraftKey, setHydratedDraftKey] = useState<string | null>(null);
  const [startRecovery, setStartRecovery] = useState<GeneratorStartRecovery | null>(null);
  const startRecoveryRef = useRef<GeneratorStartRecovery | null>(null);
  const startLock = useRef(false);
  const currentDraftKey = useRef("");
  const baseDraftKey = `admin-play-generator-draft:${clubId}:${generatorKind}`;
  const draftKey = useMemo(
    () => returnHubId || draftToken ? `${baseDraftKey}:hub:${encodeURIComponent(returnHubId || "")}:draft:${encodeURIComponent(draftToken || "legacy")}` : baseDraftKey,
    [baseDraftKey, returnHubId, draftToken]
  );
  const startRecoveryKey = `${draftKey}:start`;
  const draftHydrated = hydratedDraftKey === draftKey;
  const createdStart = startRecovery?.status === "created" ? startRecovery : null;
  const pendingStart = startRecovery?.status === "pending" ? startRecovery : null;
  const writesEnabled = status?.writes_enabled === true;

  const participantNames = useMemo(
    () => rosterNamesFromText(participantText),
    [participantText]
  );
  const automaticSetup = useMemo(
    () => recommendedGeneratorSetup(generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount, moneyball),
    [generatorKind, playFormat, targetCount, doublesCourtCount, singlesCourtCount, moneyball]
  );
  const rosterReady = participantNames.length === targetCount && (!moneyball || participantNames.every(name => Number(linkedPlayerIds[normalizeRosterName(name)] || 0) > 0));
  const mixedSetupValid = playFormat !== "doubles_singles" || (
    doublesCourtCount >= 1 &&
    singlesCourtCount >= 1 &&
    doublesCourtCount * 4 + singlesCourtCount * 2 <= targetCount
  );

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
      throw new GeneratorRequestError(String(payload?.detail || `API error (${response.status})`), response.status);
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
      if (!startRecoveryRef.current) setMessage({ kind: "error", text: error instanceof Error ? error.message : "Unable to load sessions." });
    }
  }

  useEffect(() => {
    currentDraftKey.current = draftKey;
    setHydratedDraftKey(null);
    const recovery = readGeneratorStart(startRecoveryKey);
    startRecoveryRef.current = recovery;
    setStartRecovery(recovery);
    const stored = readPlayGeneratorDraft<PreviewEvent>(draftKey) || (!draftToken ? readPlayGeneratorDraft<PreviewEvent>(baseDraftKey) : null);
    setMessage(recovery?.status === "pending" ? { kind: "error", text: "A previous start needs confirmation. Choose Retry this start to check the saved request." } : null);
    if (stored) {
      setTitle(stored.title);
      setEventFormat(generatorKind === "round_robin" ? stored.eventFormat || "standard" : "standard");
      setPlayFormat(stored.playFormat);
      setStandingsSort(stored.standingsSort || "wins");
      setScoringMode(generatorKind === "round_robin" ? stored.scoringMode || "scored" : "scored");
      setRatingMode(stored.ratingMode === "rated" ? "rated" : "unrated");
      setTargetCount(stored.targetCount);
      const mixedSetup = recommendedMixedCourtSetup(stored.targetCount);
      setDoublesCourtCount(stored.doublesCourtCount || mixedSetup.doublesCourtCount);
      setSinglesCourtCount(stored.singlesCourtCount || mixedSetup.singlesCourtCount);
      setParticipantText(stored.participantText);
      setLinkedPlayerIds(stored.linkedPlayerIds);
      setPreview(stored.preview);
      if (stored.preview && !recovery) {
        setMessage({ kind: "info", text: "Restored your unsaved schedule preview." });
      }
    } else {
      const mixedSetup = recommendedMixedCourtSetup(8);
      setTitle(`${generatorTitle(generatorKind)} ${new Date().toISOString().slice(0, 10)}`);
      setEventFormat("standard"); setPlayFormat("doubles"); setStandingsSort("wins"); setScoringMode("scored"); setRatingMode("unrated");
      setTargetCount(8); setDoublesCourtCount(mixedSetup.doublesCourtCount); setSinglesCourtCount(mixedSetup.singlesCourtCount);
      setParticipantText(""); setLinkedPlayerIds({}); setPreview(null);
    }
    setHydratedDraftKey(draftKey);
  }, [draftKey, baseDraftKey, startRecoveryKey, draftToken, generatorKind]);

  useEffect(() => {
    if (!draftHydrated || createdStart) return;
    writePlayGeneratorDraft(draftKey, {
      title,
      playFormat,
      standingsSort,
      eventFormat,
      scoringMode,
      ratingMode,
      targetCount,
      doublesCourtCount,
      singlesCourtCount,
      participantText,
      linkedPlayerIds,
      preview
    });
  }, [
    draftHydrated,
    createdStart,
    draftKey,
    title,
    playFormat,
    standingsSort,
    eventFormat,
    scoringMode,
    ratingMode,
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
  }, [accessToken, generatorKind]);

  const invalidatePreview = useCallback((): void => {
    setPreview(null);
    setMessage(null);
  }, []);

  function requestBody(): Record<string, unknown> {
    if (!rosterReady) {
      throw new Error(participantNames.length === targetCount && moneyball ? "Link or create a club profile for every Moneyball player before previewing." : `Add exactly ${targetCount} unique players before previewing.`);
    }
    if (!mixedSetupValid) {
      throw new Error("Choose a doubles and singles court mix that fits the selected player count.");
    }
    const orderedIds = participantNames.map((name) =>
      Number(linkedPlayerIds[normalizeRosterName(name)] || 0)
    );
    const allLinked = orderedIds.every((playerId) => playerId > 0);
    return {
      generator_kind: generatorKind,
      event_format: generatorKind === "round_robin" ? eventFormat : "standard",
      play_format: playFormat,
      standings_sort: moneyball ? "differential" : standingsSort,
      scoring_mode: generatorKind === "round_robin" ? scoringMode : "scored",
      rating_mode: ratingMode,
      title: title.trim(),
      participant_names: participantNames,
      player_ids: allLinked ? orderedIds : [],
      total_rounds: automaticSetup.totalRounds,
      // Zero lets the engine recalculate courts as late arrivals join later rounds.
      court_count: generatorKind === "round_robin" && playFormat !== "doubles_singles"
        ? 0
        : automaticSetup.courtCount,
      doubles_court_count: automaticSetup.doublesCourtCount,
      singles_court_count: automaticSetup.singlesCourtCount
    };
  }

  async function generatePreview(): Promise<void> {
    if (creationPending || startRecoveryRef.current) return;
    const activeDraftKey = draftKey;
    setBusy(true);
    setMessage(null);
    try {
      const body = requestBody();
      const payload = await requestJson<PreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/preview`,
        { method: "POST", body: JSON.stringify(body) }
      );
      if (currentDraftKey.current !== activeDraftKey) return;
      setPreview(payload.preview);
      setMessage({ kind: "success", text: moneyball
          ? `Fixed Moneyball schedule: ${payload.preview.rounds.length} rounds. Complete every scheduled game to decide the winner.`
          : generatorKind === "round_robin"
          ? `Starting schedule: ${payload.preview.rounds.length} rounds. Add more as needed during play.`
          : "Previewed Round 1. Later ladder rounds are generated from results."
      });
    } catch (error) {
      if (currentDraftKey.current === activeDraftKey) {
        setPreview(null);
        setMessage({ kind: "error", text: error instanceof Error ? error.message : "Unable to preview the schedule." });
      }
    } finally {
      setBusy(false);
    }
  }

  function runCreatedSession(created: CreatedGeneratorStart): void {
    if (onCreated) { onCreated(created.sessionKey); return; }
    const path = created.hubId
      ? `/admin/round-robin-hub?hub=${encodeURIComponent(created.hubId)}&add_session=${encodeURIComponent(created.sessionKey)}`
      : `/admin/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(created.sessionKey)}/rounds/${created.roundNumber}`;
    router.push(path);
    router.refresh();
  }

  function createAnotherSession(): void {
    if (startLock.current || startRecoveryRef.current?.status !== "created") return;
    const hubId = startRecoveryRef.current.hubId;
    router.push(`/admin/${generatorSlug(generatorKind)}?${hubId ? `hub=${encodeURIComponent(hubId)}&` : ""}draft=${encodeURIComponent(operationKey())}`);
    router.refresh();
  }

  async function startSession(): Promise<void> {
    if (startLock.current || creationPending || !draftHydrated) return;
    if (startRecoveryRef.current?.status === "created") { runCreatedSession(startRecoveryRef.current); return; }
    if (!preview && startRecoveryRef.current?.status !== "pending") return;
    startLock.current = true;
    setBusy(true);
    setMessage(null);
    const activeDraftKey = draftKey, recoveryKey = startRecoveryKey;
    let attempt: PendingGeneratorStart | null = startRecoveryRef.current?.status === "pending" ? startRecoveryRef.current : null;
    const recovering = Boolean(attempt);
    let confirmed: CreatedGeneratorStart | null = null;
    try {
      if (!attempt) {
        const next: PendingGeneratorStart = { status: "pending", hubId: returnHubId || null, body: {
          ...requestBody(), preview_fingerprint: preview!.previewFingerprint, expected_version: "new", idempotency_key: operationKey()
        } };
        if (!writeGeneratorStart(recoveryKey, next)) throw new Error("Your browser could not save this start for recovery. Allow browser storage and try again. Your draft has not been started.");
        attempt = next;
        startRecoveryRef.current = next; setStartRecovery(next);
      }
      const payload = await requestJson<StartResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions`,
        { method: "POST", body: JSON.stringify(attempt.body) }
      );
      const session = payload.session;
      if (!session?.session_key) throw new Error("The start response did not confirm a session key.");
      confirmed = { status: "created", sessionKey: session.session_key, roundNumber: session.current_round_number || 1, hubId: attempt.hubId };
      // If receipt storage fails, the previously stored exact request can still
      // replay the confirmed result after a reload.
      writeGeneratorStart(recoveryKey, confirmed);
      clearPlayGeneratorDraft(activeDraftKey);
      if (currentDraftKey.current === activeDraftKey) {
        startRecoveryRef.current = confirmed; setStartRecovery(confirmed);
        setMessage({ kind: "success", text: generatorKind === "round_robin" ? "Your round robin is created and ready to run." : "Your session is created and ready to run." });
      }
    } catch (error) {
      const safelyRejected = !recovering && isRejectedGeneratorStart(error);
      if (attempt && safelyRejected) writeGeneratorStart(recoveryKey, null);
      if (currentDraftKey.current === activeDraftKey) {
        if (attempt && safelyRejected) { startRecoveryRef.current = null; setStartRecovery(null); }
        setMessage({ kind: "error", text: attempt ? generatorStartFailure(error, recovering) : error instanceof Error ? error.message : "Unable to start. Your draft is saved." });
      }
    } finally {
      startLock.current = false;
      setBusy(false);
    }
    if (confirmed && currentDraftKey.current === activeDraftKey) runCreatedSession(confirmed);
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
        : `${preview.totalRounds} starting rounds - add more during play`,
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
    setMessage({ kind: "success", text: "Opened the PDF in a new tab. Your unsaved schedule remains here." });
    } catch (error) {
      closePreparedPdfWindow(pdfWindow);
      setMessage({ kind: "error", text: error instanceof Error ? error.message : "Unable to open the schedule PDF." });
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

      {message ? <p role={message.kind === "error" ? "alert" : "status"} aria-live={message.kind === "error" ? "assertive" : "polite"} style={{ color: message.kind === "error" ? "#b91c1c" : message.kind === "success" ? "#166534" : "#334155" }}>{message.text}</p> : null}

      {createdStart ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>{generatorKind === "round_robin" ? "Your round robin is ready" : "Your session is ready"}</h2>
        <p>This session has already been created. Open it to continue playing.</p>
        <div style={{ display: "flex", gap: ".6rem", flexWrap: "wrap" }}>
          <button type="button" style={primaryButton} onClick={() => runCreatedSession(createdStart)}>{createdStart.hubId ? "Run this round robin in the hub" : "Run this session"}</button>
          <button type="button" style={secondaryButton} onClick={createAnotherSession}>{generatorKind === "round_robin" ? "Create another round robin" : "Create another session"}</button>
        </div>
      </article> : null}

      {pendingStart ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>{busy ? "Starting your session…" : "Confirm this start"}</h2>
        <p>The saved request keeps the same lineup and settings. Confirm it before starting another session.</p>
        <button type="button" style={primaryButton} disabled={busy || !writesEnabled || !draftHydrated} onClick={() => void startSession()}>{busy ? "Checking…" : "Retry this start"}</button>
      </article> : null}

      {!createdStart ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Set up the session</h2>
        <p style={{ color: "#475569" }}>
          Enter players in the starting order you want. Reorder them before previewing to change
          the schedule and bye order.
        </p>
        <fieldset disabled={busy || Boolean(pendingStart)} style={{ border: 0, margin: 0, padding: 0, minWidth: 0 }}>
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
          {generatorKind === "round_robin" ? <label>
            Event format<br />
            <select aria-label="Event format" value={eventFormat} onChange={event => {
              const format = event.target.value as "standard" | "moneyball_expectations";
              setEventFormat(format);
              if (format === "moneyball_expectations") {
                setPlayFormat("doubles"); setScoringMode("scored");
                if (targetCount !== 6 && targetCount % 4 !== 0 && targetCount % 4 !== 1) setTargetCount(8);
              }
              invalidatePreview();
            }} style={inputStyle}>
              <option value="standard">Standard round robin</option>
              <option value="moneyball_expectations">Moneyball vs Expectations</option>
            </select>
            {moneyball ? <small style={{ display: "block", marginTop: ".5rem" }}>
              {targetCount} players · {automaticSetup.totalRounds} fixed rounds · games to 11, win by two. 100 pesos each; winner takes the {targetCount * 100}-peso pool.
              Select rated club players. The lineup, ratings and scheduled slots freeze before play. Substitutions disqualify both players from winning. Best performance against expectations wins; close losses can help. This is a trial format.
            </small> : null}
          </label> : null}
          <label>
            Play format
            <br />
            <select
              disabled={moneyball}
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
          <label>
            Match rating
            <br />
            <select aria-label="Match rating" value={ratingMode} onChange={event => {
              const mode = event.target.value as "rated" | "unrated";
              setRatingMode(mode);
              if (mode === "rated") setScoringMode("scored");
              invalidatePreview();
            }} style={inputStyle}>
              <option value="unrated">Unrated — stats only, no rating changes</option>
              <option value="rated">Rated — update player ratings after approval</option>
            </select>
            <small style={{ display: "block", marginTop: "0.35rem", color: "#64748b" }}>Choose before starting. This stays fixed for the session. Submit scored results afterward for admin approval.</small>
          </label>
          {generatorKind === "round_robin" ? (
            <>
              <label>
                Round scoring
                <br />
                <select
                  value={scoringMode}
                  disabled={ratingMode === "rated" || moneyball}
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
              {scoringMode === "scored" && !moneyball ? (
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
                    This primary ranking applies to the full session standings.
                  </small>
                </label>
              ) : null}
            </>
          ) : null}
        </div>

        <GeneratorRosterSetup
          apiBase={apiBase}
          clubKey={clubId}
          creationClubId={creationClubId}
          creationAccessToken={canCreatePlayer ? accessToken : undefined}
          creationDraftKey={draftKey}
          onCreationPendingChange={setCreationPending}
          disabled={busy || Boolean(pendingStart)}
          generatorKind={generatorKind}
          playFormat={playFormat}
          moneyballOnly={moneyball}
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
            disabled={busy || creationPending || Boolean(pendingStart) || !rosterReady || !mixedSetupValid}
            style={primaryButton}
          >
            {busy ? "Generating…" : "Preview matchups"}
          </button>
        </div>
        </fieldset>
      </article> : null}

      {preview && !createdStart ? (
        <article style={{ ...cardStyle, minWidth: 0 }}>
          <h2 style={{ marginTop: 0 }}>2. Preview before starting</h2>
          <p style={{ color: "#475569" }}>
            {moneyball
              ? "Review all fixed matchups and byes. Change the starting roster order above and regenerate before starting; every scheduled game must be completed to decide the winner."
              : generatorKind === "ladder"
              ? "Only Round 1 is shown. Round 2 and later are generated from saved results."
              : scoringMode === "unscored"
                ? "Review the starting schedule, matchups, and byes. Keep adding rounds for as long as you want. During play, use Round Played to move directly to the next round."
                : "Review the starting schedule, matchups, and byes. Keep adding rounds for as long as you want. Change the roster order above and regenerate when needed."}
          </p>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginBottom: "1rem" }}>
            <button type="button" onClick={downloadCsv} style={secondaryButton}>
              Download CSV
            </button>
            <button type="button" onClick={() => void downloadPdf()} style={secondaryButton}>
              Download one-sheet PDF (opens new tab)
            </button>
            {!pendingStart ? <button
              type="button"
              onClick={() => void startSession()}
              disabled={busy || creationPending || !writesEnabled || !draftHydrated}
              style={primaryButton}
            >
              {busy ? "Starting…" : returnHubId ? "Start and add to hub" : `Start ${ratingMode} session`}
            </button> : null}
          </div>

          {moneyball ? <GeneratorMoneyballExpectations rows={preview.moneyballExpectations} /> : null}

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
                          {matchFormatLabel(match, preview.playFormat)} · {match.courtLabel || `Court ${match.court || "—"}`}
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

      {!embedded ? <article style={cardStyle}>
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
                      {playFormatLabel(session.play_format)} · Round {session.current_round_number || 1}{generatorKind === "ladder" ? ` of ${session.total_rounds || "?"}` : ""}
                      {" · "}{session.status}
                    </p>
                  </div>
                  <div style={{ display: "flex", gap: ".75rem", flexWrap: "wrap", alignItems: "center" }}>
                  {session.round_robin_hub_id ? <Link href={`/admin/round-robin-hub?hub=${encodeURIComponent(session.round_robin_hub_id)}`} style={{ fontWeight: 800 }}>Run this round robin in the hub</Link> : null}
                  <Link
                    href={`/admin/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
                      session.session_key
                    )}/rounds/${session.current_round_number || 1}`}
                    style={{ fontWeight: 800 }}
                  >
                    Open session
                  </Link>
                  </div>
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
      </article> : null}
    </div>
  );
}

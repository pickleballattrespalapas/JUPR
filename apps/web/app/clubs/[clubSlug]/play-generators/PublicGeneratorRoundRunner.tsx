"use client";

import GeneratorMoneyballExpectations, { type MoneyballExpectation, type MoneyballExpectationFormat } from "@/components/GeneratorMoneyballExpectations";


import PublicControls from "@/components/PublicControls";

import { GeneratorScoreValidationError, refreshedScoreDraft, submittedScoreRows, rememberScoreSave, storedScoreSave, type PendingScoreSave } from "@/lib/generatorScoreDraft";

import GeneratorSessionDetails from "@/components/GeneratorSessionDetails";
import GeneratorSharing from "@/components/GeneratorSharing";
import { hubAutoAdvanceBoundary, hubPreparedRoundLimit } from "@/lib/hubAutoAdvanceBoundary";

import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import GeneratorRoundSchedule from "@/components/GeneratorRoundSchedule";
import GeneratorLateArrivals from "@/components/GeneratorLateArrivals";

import Link from "@/components/PublicClubLink";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ConfirmAction } from "@/components/ConfirmAction";
import GeneratorSubmission, { generatorResultLabel, type GeneratorSubmissionStatus } from "@/components/GeneratorSubmission";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import { publicLiveErrorText } from "@/lib/publicLiveErrorText";
import { swapRosterPositions } from "@/lib/playGeneratorRoster.mjs";

type GeneratorKind = "round_robin" | "ladder";
type ScoringMode = "scored" | "unscored";
type PlayFormat = "singles" | "doubles" | "doubles_singles";
type RosterAction = "add" | "remove" | "substitute" | "swap" | "reorder";

type Participant = {
  id: string;
  name: string;
  player_id?: number | null;
  roster_order?: number;
  active_from_round?: number;
  inactive_from_round?: number | null;
  inactive_rounds?: number[];
  substitutes_for?: string;
};

type MatchRow = {
  courtLabel?: string;
  id: string;
  label?: string;
  round?: number;
  court?: number;
  miniRound?: number;
  sideA?: string[];
  sideB?: string[];
  teamA?: string[];
  teamB?: string[];
  scoreA?: number | null;
  scoreB?: number | null;
  status?: string;
  sourceMatchIds?: string[];
  playFormat?: "singles" | "doubles";
};

type RoundRow = {
  number: number;
  stage?: string;
  label?: string;
  status: "preview" | "active" | "saved" | "played" | "skipped" | string;
  matches?: MatchRow[];
  courts?: Array<{
    courtNumber: number;
    participantIds?: string[];
    matches?: MatchRow[];
  }>;
  byeParticipantIds?: string[];
  warnings?: string[];
  formatCounts?: { doubles?: number; singles?: number };
  skipReason?: string | null;
};

type GeneratorEvent = {
  moneyballExpectations?: MoneyballExpectation[];
  moneyball?: MoneyballExpectationFormat;
  name: string;
  playoff?: { format: string };
  generatorKind: GeneratorKind;
  eventFormat?: "standard" | "moneyball_expectations";
  playFormat: PlayFormat;
  scoringMode?: ScoringMode;
  status: string;
  currentRoundNumber: number;
  totalRounds: number;
  courtCount?: number;
  participants: Participant[];
  rounds: RoundRow[];
  perfectRotation?: boolean;
  adaptiveFromRound?: number | null;
  rosterRevisions?: Array<Record<string, unknown>>;
  hubAssignedMatchIds?: string[];
  roundRobinHubId?: string;
  hubAutoAdvancedFromRound?: number;
  hubPreparedThroughRound?: number;
};

type GeneratorSession = {
  round_robin_hub_id?: string | null;
  results_locked?: boolean;
  submission?: GeneratorSubmissionStatus | null;
  created_at?: string;
  rating_mode?: "rated" | "unrated";
  session_key: string;
  title: string;
  status: string;
  version: string | number;
  generator_kind: GeneratorKind;
  play_format: PlayFormat;
  scoring_mode?: ScoringMode;
  current_round_number?: number | null;
  total_rounds?: number | null;
  event: GeneratorEvent;
};

type DetailResponse = {
  ok: boolean;
  session: GeneratorSession;
};

type MutationResponse = {
  ok: boolean;
  session?: GeneratorSession;
};

type SkipRoundRequest = {
  skipBody: {
    reason: string;
    edit_token: string;
    expected_version: number | string;
    idempotency_key: string;
  };
  advanceIdempotencyKey: string;
};

class ApiRequestError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

class UserFacingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UserFacingError";
  }
}

function requestFailureMessage(error: unknown, fallback: string): string {
  return error instanceof ApiRequestError || error instanceof UserFacingError
    ? error.message
    : fallback;
}

type Props = {
  apiBase: string | null;
  clubId: string;
  generatorKind: GeneratorKind;
  sessionKey: string;
  roundNumber: number;
  shared?: boolean;
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

function operationKey(action: string): string {
  const suffix =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${action}-${suffix}`;
}

function apiErrorMessage(status: number, detail?: unknown): string {
  return publicLiveErrorText(status, detail);
}

function roundStatusLabel(status: string): string {
  if (status === "saved") return "Scores saved";
  if (status === "played") return "Played";
  if (status === "skipped") return "Skipped";
  if (status === "preview") return "Preview";
  return "In progress";
}

function isUncertainRequestError(error: unknown): boolean {
  if (!(error instanceof ApiRequestError)) return true;
  if (error.status >= 500 || [408, 425, 429].includes(error.status)) return true;
  if (!error.detail || typeof error.detail !== "object") return false;
  const detail = error.detail as {
    code?: unknown;
    kind?: unknown;
    recovery_required?: unknown;
  };
  return (
    detail.code === "RECOVERY_REQUIRED" ||
    detail.kind === "uncertain" ||
    detail.recovery_required === true
  );
}

function flattenMatches(round: RoundRow | null): MatchRow[] {
  if (!round) return [];
  if (round.matches?.length) return round.matches;
  return (round.courts || []).flatMap((court) => court.matches || []);
}

function participantMap(event: GeneratorEvent): Map<string, Participant> {
  return new Map((event.participants || []).map((participant) => [participant.id, participant]));
}

function sideIds(match: MatchRow, side: "A" | "B"): string[] {
  return side === "A"
    ? match.sideA || match.teamA || []
    : match.sideB || match.teamB || [];
}

function sideLabel(match: MatchRow, side: "A" | "B", participants: Map<string, Participant>): string {
  return sideIds(match, side)
    .map((id) => participants.get(String(id))?.name || String(id))
    .join(" / ");
}

function scoreKey(matchId: string, side: "a" | "b"): string {
  return `${matchId}:${side}`;
}

function roundPath(kind: GeneratorKind, clubSlug: string, sessionKey: string, round: number): string {
  return `/clubs/${clubSlug}/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;
}

function standingsPath(clubSlug: string, sessionKey: string): string {
  return `/clubs/${clubSlug}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/standings`;
}

function roundStandings(
  round: RoundRow,
  participants: Map<string, Participant>
): Array<{
  participantId: string;
  name: string;
  wins: number;
  losses: number;
  pointsFor: number;
  pointsAgainst: number;
  differential: number;
}> {
  const stats = new Map<
    string,
    {
      participantId: string;
      name: string;
      wins: number;
      losses: number;
      pointsFor: number;
      pointsAgainst: number;
      differential: number;
    }
  >();
  for (const match of flattenMatches(round)) {
    if (match.status === "not_played" || match.scoreA == null || match.scoreB == null) continue;
    const scoreA = Number(match.scoreA);
    const scoreB = Number(match.scoreB);
    for (const id of sideIds(match, "A")) {
      const row = stats.get(id) || {
        participantId: id,
        name: participants.get(id)?.name || id,
        wins: 0,
        losses: 0,
        pointsFor: 0,
        pointsAgainst: 0,
        differential: 0
      };
      row.pointsFor += scoreA;
      row.pointsAgainst += scoreB;
      row.differential += scoreA - scoreB;
      if (scoreA > scoreB) row.wins += 1;
      else row.losses += 1;
      stats.set(id, row);
    }
    for (const id of sideIds(match, "B")) {
      const row = stats.get(id) || {
        participantId: id,
        name: participants.get(id)?.name || id,
        wins: 0,
        losses: 0,
        pointsFor: 0,
        pointsAgainst: 0,
        differential: 0
      };
      row.pointsFor += scoreB;
      row.pointsAgainst += scoreA;
      row.differential += scoreB - scoreA;
      if (scoreB > scoreA) row.wins += 1;
      else row.losses += 1;
      stats.set(id, row);
    }
  }
  return [...stats.values()].sort(
    (left, right) =>
      right.wins - left.wins ||
      right.differential - left.differential ||
      right.pointsFor - left.pointsFor ||
      left.name.localeCompare(right.name)
  );
}

export default function GeneratorRoundRunner({
  apiBase,
  clubId,
  generatorKind,
  sessionKey,
  roundNumber,
  shared = false
}: Props) {
  function sessionRoundPath(kind: GeneratorKind, slug: string, key: string, number: number): string {
    return shared ? `/clubs/${slug}/round-robin-generator/shared/${encodeURIComponent(key)}/rounds/${number}` : roundPath(kind, slug, key, number);
  }
  function sessionStandingsPath(slug: string, key: string): string {
    return shared ? `/clubs/${slug}/round-robin-generator/shared/${encodeURIComponent(key)}/standings` : standingsPath(slug, key);
  }
  function sessionVersion(value: string | number): string | number { return shared ? String(value) : Number(value); }
  const router = useRouter();
  const [editToken, setEditToken] = useState("");
  const [session, setSession] = useState<GeneratorSession | null>(null);
  const [scores, setScores] = useState<Record<string, string>>({});
  const [editingSavedScores, setEditingSavedScores] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [skipReason, setSkipReason] = useState("");
  const [rosterAction, setRosterAction] = useState<RosterAction>("add");
  const [selectedParticipant, setSelectedParticipant] = useState("");
  const [firstSwapParticipant, setFirstSwapParticipant] = useState("");
  const [secondSwapParticipant, setSecondSwapParticipant] = useState("");
  const [newPlayerName, setNewPlayerName] = useState("");
  const [newPlayerId, setNewPlayerId] = useState("");
  const [substituteScope, setSubstituteScope] = useState<"round" | "rest">("rest");
  const [rosterOrder, setRosterOrder] = useState<string[]>([]);
  const skipDestinationRef = useRef<number | "completed" | null>(null);
  const context = JSON.stringify([apiBase, clubId, sessionKey, roundNumber, generatorKind, shared]);
  const mountedRef = useRef(false);
  const contextRef = useRef(context);
  const generationRef = useRef(0);
  const sessionLoadRef = useRef<AbortController | null>(null);
  const appliedSessionRef = useRef<GeneratorSession | null>(null);
  const writeInFlightRef = useRef(false);
  const scoreSaveInFlightRef = useRef(false);
  const pendingScoreSaveRef = useRef<PendingScoreSave | null>(null);
  const [pendingScoreSave, setPendingScoreSave] = useState<PendingScoreSave | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    contextRef.current = context;
    appliedSessionRef.current = null;
    const pending = storedScoreSave(context);
    pendingScoreSaveRef.current = pending;
    setPendingScoreSave(pending);
    setSession(null);
    setScores({});
    setEditingSavedScores(false);
    setSkipReason("");
    setSelectedParticipant("");
    setFirstSwapParticipant("");
    setSecondSwapParticipant("");
    setNewPlayerName("");
    setNewPlayerId("");
    setRosterOrder([]);
    setMessage(null);
    writeInFlightRef.current = false;
    scoreSaveInFlightRef.current = false;
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      sessionLoadRef.current?.abort();
      sessionLoadRef.current = null;
    };
  }, [context]);

  function currentContext(generation = generationRef.current): boolean {
    return mountedRef.current && contextRef.current === context && generationRef.current === generation;
  }

  function retainScoreSave(request: PendingScoreSave | null, expected?: PendingScoreSave): void {
    pendingScoreSaveRef.current = request;
    setPendingScoreSave(request);
    rememberScoreSave(context, request, expected);
  }


  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) {
      throw new ApiRequestError(
        "This play tool is temporarily unavailable. Please try again.",
        503,
        "local_configuration"
      );
    }
    const generation = generationRef.current;
    const writing = Boolean(options?.method && options.method !== "GET");
    if (writing && writeInFlightRef.current) throw new ApiRequestError("Another action is still being saved.", 409, "local_in_flight");
    const pending = pendingScoreSaveRef.current;
    if (writing && pending && (pending.path !== path || pending.body !== options?.body)) {
      throw new ApiRequestError("Retry the pending score save before starting another action.", 409, "pending_score_save");
    }
    if (writing) {
      writeInFlightRef.current = true;
      sessionLoadRef.current?.abort();
      sessionLoadRef.current = null;
    }
    try {
      const headers = new Headers(options?.headers);
      if (options?.body) headers.set("Content-Type", "application/json");
      const response = await fetch(apiUrl(apiBase, shared ? path.replace("/play-generators/", "/shared-play-generators/") : path), {
        ...options,
        headers,
        cache: "no-store"
      });
      const payload = await response.json().catch(() => null);
      if (!currentContext(generation)) throw new DOMException("Request context changed", "AbortError");
      if (options?.signal?.aborted) throw new DOMException("Request aborted", "AbortError");
      if (!response.ok) {
        if (shared && response.status === 403) {
          sessionStorage.removeItem(`shared-generator-edit:${clubId}:${sessionKey}`);
          setEditToken("");
        }
        const detail = payload?.detail;
        throw new ApiRequestError(apiErrorMessage(response.status, detail), response.status, detail);
      }
      return payload as T;
    } finally {
      if (writing && currentContext(generation)) writeInFlightRef.current = false;
    }
  }

  function applySession(next: GeneratorSession, preserveDraftScores = false): void {
    if (!currentContext()) return;
    const previous = appliedSessionRef.current;
    const previousRound = previous?.event.rounds.find(row => row.number === roundNumber) || null;
    const requestedRound = next.event.rounds.find(row => row.number === roundNumber) || null;
    appliedSessionRef.current = next;
    setSession(next);
    if (next.event.eventFormat === "moneyball_expectations") setRosterAction("substitute");
    if (!preserveDraftScores) setEditingSavedScores(false);
    setScores(current => refreshedScoreDraft(flattenMatches(previousRound), flattenMatches(requestedRound), current, preserveDraftScores));
    const ordered = [...next.event.participants]
      .sort(
        (left, right) =>
          Number(left.roster_order || 0) - Number(right.roster_order || 0)
      )
      .map((row) => row.id);
    setRosterOrder(ordered);
  }

  async function submitResults(organizerName: string, matchDate: string): Promise<void> {
    const generation = generationRef.current;
    if (!session) return;
    const payload = await requestJson<MutationResponse>(
      `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/submit`,
      { method: "POST", body: JSON.stringify({ edit_token: editToken, idempotency_key: `generator-submit:${sessionKey}`, expected_version: session.version, organizer_name: organizerName, match_date: matchDate }) }
    );
    if (!currentContext(generation)) throw new DOMException("Request context changed", "AbortError");
    if (!payload.session) throw new Error("Could not confirm submission. Please try again.");
    applySession(payload.session);
  }

  async function loadSession(): Promise<boolean> {
    sessionLoadRef.current?.abort();
    const controller = new AbortController();
    sessionLoadRef.current = controller;
    const generation = generationRef.current;
    const isCurrentLoad = () => currentContext(generation) && sessionLoadRef.current === controller && !controller.signal.aborted;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<DetailResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}`,
        { signal: controller.signal }
      );
      if (!isCurrentLoad()) return false;
      if (payload.session.generator_kind !== generatorKind) {
        throw new UserFacingError("This link opens in the other play generator.");
      }
      applySession(payload.session, true);
      const hasRequested = payload.session.event.rounds.some((row) => row.number === roundNumber)
        && (!payload.session.round_robin_hub_id || roundNumber <= hubPreparedRoundLimit(payload.session.event));
      if (!hasRequested) {
        router.replace(
          sessionRoundPath(
            generatorKind,
            clubId,
            sessionKey,
            payload.session.current_round_number || 1
          )
        );
      }
      return true;
    } catch (error) {
      if (isCurrentLoad()) {
        setMessage(requestFailureMessage(error, "We couldn’t load this session. Please try again."));
      }
      return false;
    } finally {
      if (isCurrentLoad()) {
        sessionLoadRef.current = null;
        setBusy(false);
      }
    }
  }

  useEffect(() => {
    const storageKey = `${shared ? "shared-generator-edit" : "public-generator-edit"}:${clubId}:${sessionKey}`;
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";
    if (discovered) {
      sessionStorage.setItem(storageKey, discovered);
    }
    setEditToken(discovered);
    if (hash.has("edit")) {
      window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);
    }
    void loadSession();
    return () => {
      sessionLoadRef.current?.abort();
      sessionLoadRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [context]);

  const event = session?.event || null;
  const participants = useMemo(
    () => (event ? participantMap(event) : new Map<string, Participant>()),
    [event]
  );
  const round =
    event?.rounds.find((row) => Number(row.number) === Number(roundNumber)) || null;
  const matches = flattenMatches(round);
  const isCurrent =
    Boolean(event) && Number(event?.currentRoundNumber || 1) === Number(roundNumber);
  const visibleRoundLimit = session?.round_robin_hub_id ? hubPreparedRoundLimit(event) : Number(event?.currentRoundNumber || 1);
  const scoringMode: ScoringMode = session?.scoring_mode || event?.scoringMode || "scored";
  const scoredSession = scoringMode === "scored";
  const resultsLocked = Boolean(session?.submission || session?.results_locked);
  const canManage = Boolean(editToken) && !resultsLocked;
  const canScore = canManage;
  const canEditRound =
    canScore &&
    session?.status === "active" &&
    (!session?.round_robin_hub_id || isCurrent) &&
    (isCurrent || (generatorKind === "round_robin" && !event?.playoff)) &&
    round?.status === "active";
  const canEditSavedScores =
    canScore && scoredSession && round?.status === "saved" &&
    (!session?.round_robin_hub_id || roundNumber <= Number(event?.currentRoundNumber || 1)) &&
    (session?.status === "active" || session?.status === "completed") &&
    (isCurrent || (generatorKind === "round_robin" && !event?.playoff));
  const editingScores = canEditSavedScores && editingSavedScores;
  const canClearSavedRound = canScore && generatorKind === "round_robin" && round?.status === "saved" && !event?.playoff && event?.eventFormat !== "moneyball_expectations";
  const canReopenRound = canManage && !session?.round_robin_hub_id && !event?.playoff && generatorKind === "round_robin" && round?.status === "skipped";
  const hasOtherOpenRound = event?.rounds.some(row => row.status === "active" && row.number !== event.currentRoundNumber) ?? false;
  const canFinishSession = canManage && (event?.eventFormat !== "moneyball_expectations" || event.rounds.every(row => row.status === "saved")) && !event?.playoff && generatorKind === "round_robin" && session?.status === "active" && isCurrent && !hasOtherOpenRound && (!session?.round_robin_hub_id || ((["saved", "played", "skipped"].includes(round?.status || "") || hubAutoAdvanceBoundary(event) !== null) && !event?.hubAssignedMatchIds?.length));
  const draftScoreCount = scoredSession ? matches.reduce((count, match) => count + (["a", "b"] as const)
    .filter(side => {
      const saved = side === "a" ? match.scoreA : match.scoreB;
      return (scores[scoreKey(match.id, side)] ?? "") !== (saved == null ? "" : String(saved));
    }).length, 0) : 0;
  const anyDraftScore = draftScoreCount > 0;
  const results = round ? roundStandings(round, participants) : [];
  const byeNames = (round?.byeParticipantIds || [])
    .map((id) => participants.get(id)?.name || id)
    .join(", ");

  async function runMutation(path: string, body: Record<string, unknown>, preserveDraftScores = false): Promise<GeneratorSession> {
    const generation = generationRef.current;
    const payload = await requestJson<MutationResponse>(path, {
      method: "POST",
      body: JSON.stringify({
        ...body,
        edit_token: editToken,
        expected_version: sessionVersion(session?.version || 1),
        idempotency_key: operationKey(String(body.action || path.split("/").pop() || "mutation"))
      })
    });
    if (!currentContext(generation)) throw new DOMException("Request context changed", "AbortError");
    if (!payload.session) throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
    applySession(payload.session, preserveDraftScores);
    return payload.session;
  }

  async function saveRound(): Promise<void> {
    if (!session || !round || !canScore || scoreSaveInFlightRef.current || writeInFlightRef.current) return;
    const generation = generationRef.current;
    let request = pendingScoreSaveRef.current;
    if (!request) {
      try {
        const requiredWinnerIds = (event?.rounds || []).flatMap(row => flattenMatches(row).flatMap(match => match.sourceMatchIds || []));
        const scoreRows = submittedScoreRows(matches, scores, event?.eventFormat === "moneyball_expectations", requiredWinnerIds);
        request = { context, path: `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/rounds/${roundNumber}/scores`,
          body: JSON.stringify({ scores: scoreRows, edit_token: editToken, expected_version: sessionVersion(session.version), idempotency_key: operationKey("scores") }) };
      } catch (error) {
        setMessage(error instanceof GeneratorScoreValidationError ? error.message : "Check the scores before saving.");
        return;
      }
      retainScoreSave(request);
    }
    scoreSaveInFlightRef.current = true;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<MutationResponse>(request.path, { method: "PATCH", body: request.body });
      if (!currentContext(generation)) return;
      if (!payload.session) throw new Error("Scores saved without a refreshed session.");
      retainScoreSave(null, request);
      applySession(payload.session);
      setMessage(`Round ${roundNumber} scores saved.`);
    } catch (error) {
      if (!currentContext(generation)) return;
      if (isUncertainRequestError(error)) {
        setMessage("The score save could not be confirmed. Retry score save to verify the exact same request before making another change.");
      } else {
        retainScoreSave(null, request);
        if (error instanceof ApiRequestError && error.status === 409) {
          const refreshed = await loadSession();
          if (!currentContext(generation)) return;
          setMessage(refreshed ? "This session changed. Your score entries are preserved. Review the current games before saving again." : "This session changed. Your score entries are preserved. Refresh and review the current games before saving again.");
        } else setMessage(requestFailureMessage(error, "We couldn’t save the scores. Please try again."));
      }
    } finally {
      if (currentContext(generation)) {
        scoreSaveInFlightRef.current = false;
        setBusy(false);
      }
    }
  }

  async function markRoundPlayed(): Promise<void> {
    const generation = generationRef.current;
    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;
    setBusy(true);
    setMessage(null);
    try {
      const playedPayload = await requestJson<MutationResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/played`,
        {
          method: "POST",
          body: JSON.stringify({
            edit_token: editToken,
            expected_version: sessionVersion(session.version),
            idempotency_key: operationKey("played")
          })
        }
      );
      if (!playedPayload.session) {
        throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
      }
      applySession(playedPayload.session);
      if (playedPayload.session.status === "completed") {
        setMessage("Session completed.");
        router.refresh();
        return;
      }
      const nextRound = playedPayload.session.current_round_number || roundNumber + 1;
      router.push(sessionRoundPath(generatorKind, clubId, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t update the round. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }


  async function executeSkipRound(request: SkipRoundRequest): Promise<ActionCompletion> {
    const generation = generationRef.current;
    setBusy(true);
    setMessage(null);
    let skipCommitted = false;
    try {
      const payload = await requestJson<MutationResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/skip`,
        {
          method: "POST",
          body: JSON.stringify(request.skipBody)
        }
      );
      if (!payload.session) throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
      skipCommitted = true;
      applySession(payload.session);
      if (generatorKind === "round_robin" && !scoredSession && isCurrent) {
        const advancedPayload = await requestJson<MutationResponse>(
          `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
            sessionKey
          )}/advance`,
          {
            method: "POST",
            body: JSON.stringify({
              edit_token: request.skipBody.edit_token,
              expected_version: sessionVersion(payload.session.version),
              idempotency_key: request.advanceIdempotencyKey
            })
          }
        );
        if (!advancedPayload.session) throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
        applySession(advancedPayload.session);
        if (advancedPayload.session.status === "completed") {
          setMessage("Session completed.");
          skipDestinationRef.current = "completed";
          return actionSuccess(
            "Round skipped and session completed",
            `Round ${roundNumber} was skipped${draftScoreCount ? ` and ${draftScoreCount} unsaved score ${draftScoreCount === 1 ? "entry was" : "entries were"} discarded` : ""}. The session is complete.`
          );
        } else {
          const nextRound = advancedPayload.session.current_round_number || roundNumber + 1;
          skipDestinationRef.current = nextRound;
          return actionSuccess(
            "Round skipped",
            `Round ${roundNumber} was skipped${draftScoreCount ? ` and ${draftScoreCount} unsaved score ${draftScoreCount === 1 ? "entry was" : "entries were"} discarded` : ""}. Round ${nextRound} is now current.`
          );
        }
      }
      const successMessage = `Round ${roundNumber} was skipped${draftScoreCount ? ` and ${draftScoreCount} unsaved score ${draftScoreCount === 1 ? "entry was" : "entries were"} discarded` : ""}.`;
      setMessage(successMessage);
      return actionSuccess("Round skipped", successMessage);
    } catch (error) {
      if (!currentContext(generation)) throw error;
      const errorMessage = requestFailureMessage(error, "We couldn’t skip the round. Please try again.");
      if (skipCommitted || isUncertainRequestError(error)) {
        const recoveryMessage = skipCommitted
          ? "The round was skipped, but we couldn’t load the next one. Try again before making another change."
          : "We couldn’t confirm the update. Try again before making another change.";
        setMessage(recoveryMessage);
        return actionUncertain(
          "Update not confirmed",
          recoveryMessage,
          request.skipBody.idempotency_key,
          "Try again",
          () => executeSkipRound(request),
          false
        );
      }
      setMessage(errorMessage);
      throw error;
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }

  function skipRound(): Promise<ActionCompletion> {
    if (round?.status === "saved" && !skipReason.trim()) throw new UserFacingError("Enter a reason before clearing saved results.");
    if (!session || !round) throw new UserFacingError("Reload the session before skipping this round.");
    skipDestinationRef.current = null;
    return executeSkipRound({
      skipBody: {
        reason: skipReason,
        edit_token: editToken,
        expected_version: sessionVersion(session.version),
        idempotency_key: operationKey("skip")
      },
      advanceIdempotencyKey: operationKey("advance-after-skip")
    });
  }

  function acknowledgeSkip(): void {
    const destination = skipDestinationRef.current;
    skipDestinationRef.current = null;
    if (typeof destination === "number") {
      router.push(sessionRoundPath(generatorKind, clubId, sessionKey, destination));
    }
    if (destination !== null) router.refresh();
  }

  async function openScheduledRound(number: number): Promise<void> {
    const generation = generationRef.current;
    if (!session || !event || busy || editingScores || (canEditRound && anyDraftScore)) return;
    const target = event.rounds.find(row => row.number === number);
    if (!target || (number === roundNumber && isCurrent)) return;
    if (session.round_robin_hub_id && number > hubPreparedRoundLimit(event)) {
      setMessage("This round has not been prepared by the hub yet. Refresh to see the latest matchups.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      if (canManage && !session.round_robin_hub_id && !event.playoff && ["preview", "active", "skipped"].includes(target.status)) {
        await runMutation(`/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/rounds/${number}/select`, {});
      }
      router.push(sessionRoundPath(generatorKind, clubId, sessionKey, number));
      router.refresh();
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t open that round. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }

  async function reopenRound(): Promise<void> {
    const generation = generationRef.current;
    if (!canReopenRound || busy) return;
    setBusy(true);
    setMessage(null);
    try {
      await runMutation(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/rounds/${roundNumber}/reopen`,
        {}
      );
      setMessage(`Round ${roundNumber} is ready to play. ${scoredSession ? "Enter the scores below." : "Mark it played when finished."}`);
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t reopen this round. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }

  async function finishSession(): Promise<ActionCompletion> {
    const generation = generationRef.current;
    if (!canFinishSession || busy) throw new Error("Finish the open rounds before completing this session.");
    setBusy(true);
    setMessage(null);
    try {
      await runMutation(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/complete`,
        {}
      );
      setMessage("Session complete. Your saved results are ready to review.");
      return actionSuccess("Session complete", "Your saved results are ready to review.");
    } catch (error) {
      if (!currentContext(generation)) throw error;
      setMessage(requestFailureMessage(error, "We couldn’t finish the session. Please try again."));
      throw error;
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }

  async function advanceRound(): Promise<void> {
    const generation = generationRef.current;
    if (!session) return;
    setBusy(true);
    setMessage(null);
    try {
      const next = await runMutation(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/advance`,
        {}
      );
      if (next.status === "completed") {
        setMessage("Session complete. You can review the saved matches.");
        if (next.event.eventFormat === "moneyball_expectations") router.push(sessionStandingsPath(clubId, sessionKey));
        router.refresh();
        return;
      }
      const nextRound = next.current_round_number || roundNumber + 1;
      router.push(sessionRoundPath(generatorKind, clubId, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t continue to the next round. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }

  function moveRoster(index: number, direction: -1 | 1): void {
    setRosterOrder((current) => {
      const next = [...current];
      const target = index + direction;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  async function saveRosterChange(): Promise<void> {
    const generation = generationRef.current;
    if (!session) return;
    setBusy(true);
    setMessage(null);
    try {
      const body: Record<string, unknown> = {
        action: rosterAction === "swap" ? "reorder" : rosterAction
      };
      if (rosterAction === "reorder") {
        body.roster_order = rosterOrder;
      } else if (rosterAction === "swap") {
        body.roster_order = swapRosterPositions(
          rosterOrder,
          firstSwapParticipant,
          secondSwapParticipant
        );
      } else if (rosterAction === "remove") {
        body.participant_id = selectedParticipant;
      } else if (rosterAction === "add") {
        body.name = newPlayerName;
        body.player_id = newPlayerId ? Number(newPlayerId) : null;
      } else {
        body.participant_id = selectedParticipant;
        body.name = newPlayerName;
        body.player_id = newPlayerId ? Number(newPlayerId) : null;
        body.substitute_scope = substituteScope;
      }
      const next = await runMutation(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/roster`,
        body,
        true
      );
      setSelectedParticipant("");
      setFirstSwapParticipant("");
      setSecondSwapParticipant("");
      setNewPlayerName("");
      setNewPlayerId("");
      setMessage(next.event.eventFormat === "moneyball_expectations"
        ? "Substitution applied. Both players are ineligible to win; expectations remain fixed."
        : next.round_robin_hub_id
        ? "Players updated. Games already assigned or completed stay in place; waiting matchups use the updated roster."
        : generatorKind === "round_robin" || roundNumber < next.event.totalRounds
        ? `Players updated for Round ${roundNumber + 1}. Current games and score entries stay in place.`
        : "This is the final round. Added arrivals can play together on a spare court below.");
      const current = next.current_round_number || roundNumber;
      if (current !== roundNumber) {
        router.push(sessionRoundPath(generatorKind, clubId, sessionKey, current));
      } else {
        router.refresh();
      }
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t update the players. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }
  async function startArrivalGame(participantIds: string[], court: number): Promise<void> {
    const generation = generationRef.current;
    setBusy(true);
    setMessage(null);
    try {
      await runMutation(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/roster`,
        { action: "seat_arrivals", participant_ids: participantIds, court_number: court },
        true
      );
      setMessage(session?.round_robin_hub_id
        ? "Late-arrival game added to the hub queue. Assign its court from the court board."
        : `Late arrivals are playing on Court ${court}. All existing games and score entries stay in place.`);
    } catch (error) {
      if (!currentContext(generation)) return;
      setMessage(requestFailureMessage(error, "We couldn’t add the late-arrival game. Please try again."));
    } finally {
      if (currentContext(generation)) setBusy(false);
    }
  }


  if (!session || !event || !round) {
    return (
      <article style={cardStyle}>
        <h1>{generatorTitle(generatorKind)}</h1>
        <p>{busy ? "Loading session…" : message || "Session unavailable."}</p>
        <Link href={`/clubs/${clubId}/${generatorSlug(generatorKind)}`}>Back to generator</Link>
      </article>
    );
  }

  const previousRound = roundNumber > 1 ? roundNumber - 1 : null;
  const nextExisting = event.rounds.some((row) => row.number === roundNumber + 1)
    ? roundNumber + 1
    : null;
  const doublesGames = round.formatCounts?.doubles ?? matches.filter(
    (match) => matchFormatLabel(match, event.playFormat) === "Doubles"
  ).length;
  const singlesGames = round.formatCounts?.singles ?? matches.filter(
    (match) => matchFormatLabel(match, event.playFormat) === "Singles"
  ).length;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.4rem" }}>
          <Link href={`/clubs/${clubId}/${generatorSlug(generatorKind)}`}>
            ← {generatorTitle(generatorKind)}
          </Link>
        </p>
        <h1 style={{ margin: "0 0 0.4rem" }}>{session.title}</h1>
        <p style={{ margin: 0, color: "#475569" }}>
          {playFormatLabel(session.play_format)} · Round {roundNumber}{generatorKind === "ladder" ? ` of ${event.totalRounds}` : ""} · {scoredSession ? "Scores on" : "Scores off"} · {roundStatusLabel(round.status)} · {session.submission?.status !== "approved" ? `${session.rating_mode === "rated" ? "Rated" : "Unrated"} · ` : ""}{generatorResultLabel(session.submission)}
        </p>
        {!editToken ? <p style={{ color: "#64748b" }}>Only the organizer can enter scores or change players.</p> : null}
              <GeneratorSessionDetails key={sessionKey} apiBase={apiBase} clubId={clubId} sessionKey={sessionKey} session={session} ownerToken={shared ? undefined : editToken} locked={resultsLocked} disabled={busy || Boolean(pendingScoreSave) || editingScores || (canEditRound && anyDraftScore)} scored={scoredSession} onSaved={applySession} />
      </article>

      <PublicControls navigation resetKey={String(roundNumber)} context={`Round ${roundNumber} · ${roundStatusLabel(round.status)}`}>
{generatorKind === "round_robin" && (shared || editToken) ? <GeneratorSharing apiBase={apiBase} clubId={clubId} sessionKey={sessionKey} version={session.version} ownerToken={shared ? undefined : editToken} shared={shared} locked={resultsLocked} disabled={busy || Boolean(pendingScoreSave) || editingScores || (canEditRound && anyDraftScore)} onChanged={async () => { await loadSession(); }} /> : null}

      {session.round_robin_hub_id ? <aside style={{ ...cardStyle, background: "#eff6ff" }}>
        <strong>Courts are managed in the Round Robin Hub.</strong>
        <p style={{ marginBottom: 0 }}>The hub releases prepared matches as players finish their earlier games. Standings update in round order after each whole round is complete.</p>
        {roundNumber > event.currentRoundNumber ? <p style={{ marginBottom: 0 }}>View only here. The hub controls when these matches can play and records their results.</p> : null}
        {round.status === "active" && matches.some(match => match.scoreA != null && match.scoreB != null) ? <p style={{ marginBottom: 0 }}>
          Round {roundNumber}: {matches.filter(match => match.scoreA != null && match.scoreB != null).length} of {matches.length} matches complete — scores saved, standings pending.
        </p> : null}
      </aside> : null}

      {generatorKind === "round_robin" && !event.playoff && event.eventFormat !== "moneyball_expectations" ? <p>The rounds shown are a preview, not a limit. Continue to create additional rounds when you need them.</p> : null}

      {generatorKind === "round_robin" && !event.playoff ? (
        <GeneratorRoundSchedule rounds={session.round_robin_hub_id ? event.rounds.filter(row => row.number <= visibleRoundLimit) : event.rounds} participants={event.participants}
          selectedRound={roundNumber} currentRound={event.currentRoundNumber}
          disabled={busy || Boolean(pendingScoreSave) || editingScores || (canEditRound && anyDraftScore)}
          unsaved={editingScores || (canEditRound && anyDraftScore)}
          perfect={Boolean(event.perfectRotation)} adaptive={Boolean(event.adaptiveFromRound)} moneyball={event.eventFormat === "moneyball_expectations"}
          hubManaged={Boolean(session.round_robin_hub_id)}
          onOpen={number => void openScheduledRound(number)} />
      ) : null}

      </PublicControls>

      {scoredSession && !shared && (session.status === "completed" || session.submission) ? <GeneratorSubmission ratingMode={session.rating_mode || "unrated"} submission={session.submission} canSubmit={!pendingScoreSave && session.status === "completed" && Boolean(editToken) && !editingScores} defaultDate={session.created_at} onSubmit={submitResults} /> : null}

      {event.eventFormat === "moneyball_expectations" ? <details style={{ ...cardStyle, minWidth: 0 }}>
        <summary style={{ cursor: "pointer", fontWeight: 800 }}>View full-event Moneyball expectations</summary>
        <GeneratorMoneyballExpectations rows={event.moneyballExpectations} locked format={event.moneyball} />
      </details> : null}

      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Session complete</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>Review the saved session history below.</p>
          {generatorKind === "round_robin" && canManage && !session.round_robin_hub_id && !editingScores && !event.playoff && event.eventFormat !== "moneyball_expectations" ? (
            <button type="button" onClick={() => void advanceRound()} disabled={busy || Boolean(pendingScoreSave)} style={{ ...primaryButton, marginTop: "1rem" }}>
              {busy ? "Continuing…" : "Keep playing"}
            </button>
          ) : null}
        </article>
      ) : null}

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>{round.label || `Round ${roundNumber}`}</h2>
            {event.playFormat === "doubles_singles" ? (
              <p style={{ color: "#475569" }}>
                {doublesGames} doubles game{doublesGames === 1 ? "" : "s"} · {singlesGames} singles game{singlesGames === 1 ? "" : "s"}
              </p>
            ) : null}
            {generatorKind === "ladder" ? (
              <p style={{ color: "#475569" }}>
                Later rounds are created only after this round is saved or skipped.
              </p>
            ) : null}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {generatorKind === "round_robin" && scoredSession ? (
              <Link href={sessionStandingsPath(clubId, sessionKey)} style={secondaryButton}>
                Standings
              </Link>
            ) : null}
            {!isCurrent ? (
              <Link href={sessionRoundPath(generatorKind, clubId, sessionKey, event.currentRoundNumber)} style={secondaryButton}>
                Current round
              </Link>
            ) : null}
            {previousRound ? (
              <Link href={sessionRoundPath(generatorKind, clubId, sessionKey, previousRound)} style={secondaryButton}>
                Previous round
              </Link>
            ) : null}
            {nextExisting && nextExisting <= visibleRoundLimit ? (
              <Link href={sessionRoundPath(generatorKind, clubId, sessionKey, nextExisting)} style={secondaryButton}>
                Next round
              </Link>
            ) : null}
          </div>
        </div>

        {byeNames ? (
          <p style={{ padding: "0.6rem", background: "#fff7ed", borderRadius: "8px" }}>
            <strong>Byes:</strong> {byeNames}
          </p>
        ) : null}

        {canEditSavedScores ? (
          <div style={{ marginBottom: "1rem" }}>
            {editingScores ? (
              <>
                <p>Update the scores below, then save your changes.</p>
                <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
                  <button type="button" onClick={() => void saveRound()} disabled={busy} style={primaryButton}>
                    {busy ? "Saving…" : pendingScoreSave ? "Retry score save" : "Save score changes"}
                  </button>
                  <button type="button" onClick={() => applySession(session)} disabled={busy || Boolean(pendingScoreSave)} style={secondaryButton}>
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <button type="button" onClick={() => { applySession(session); setMessage(null); setEditingSavedScores(true); }} disabled={busy || Boolean(pendingScoreSave)} style={secondaryButton}>
                Edit scores
              </button>
            )}
          </div>
        ) : null}

        {scoredSession && (canEditRound || editingScores) ? <p>
          {event.eventFormat === "moneyball_expectations" ? "Enter every score before submitting the Moneyball rotation." : "When you submit, leave both scores blank for a game that was not played. Enter both scores for every played game. Blank draft fields are not saved until you submit."}
        </p> : null}
        {pendingScoreSave ? <aside role="status">
          <p>A score save needs confirmation. Retry the same save before making another change.</p>
          {!canEditRound && !editingScores ? <button type="button" disabled={busy} onClick={() => void saveRound()} style={primaryButton}>{busy ? "Saving…" : "Retry score save"}</button> : null}
        </aside> : null}
        {canClearSavedRound && !editingScores ? <div style={{ marginBottom: "1rem" }}>
          <input aria-label="Reason for clearing saved round" value={skipReason} onChange={event_ => setSkipReason(event_.target.value)} placeholder="Reason for clearing saved results" disabled={busy || Boolean(pendingScoreSave)} style={inputStyle} />
          <ConfirmAction triggerLabel="Clear results and skip round" title={`Clear Round ${roundNumber} results?`} description="This removes the results from this round and marks it skipped. Later rounds and their results remain in place. The reason is recorded in the audit history." confirmLabel="Clear results and skip round" confirmationText="SKIP ROUND" tone="danger" disabled={busy || Boolean(pendingScoreSave) || !skipReason.trim()} busy={busy} onConfirm={skipRound} onAcknowledge={acknowledgeSkip} />
        </div> : null}

        <div style={{ display: "grid", gap: "0.75rem" }}>
          {matches.map((match) => {
            const editable = (canEditRound || editingScores) && scoredSession;
            return (
              <article
                key={match.id}
                style={{
                  border: "1px solid #e2e8f0",
                  borderRadius: "12px",
                  padding: "0.8rem",
                  background: "#f8fafc"
                }}
              >
                <p style={{ margin: "0 0 0.4rem", color: "#64748b" }}>
                  {matchFormatLabel(match, event.playFormat)} · {match.courtLabel || `Court ${match.court || "—"}`}
                  {match.miniRound ? ` · Game ${match.miniRound}` : ""}
                  {match.label ? ` · ${match.label}` : ""}
                </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: editable
                      ? "minmax(0, 1fr) 4.5rem 4.5rem minmax(0, 1fr)"
                      : "minmax(0, 1fr) auto minmax(0, 1fr)",
                    gap: "0.65rem",
                    alignItems: "center"
                  }}
                >
                  <strong>{sideLabel(match, "A", participants)}</strong>
                  {editable ? (
                    <>
                      <input
                        value={scores[scoreKey(match.id, "a")] ?? ""}
                        onChange={(event_) =>
                          setScores((current) => ({
                            ...current,
                            [scoreKey(match.id, "a")]: event_.target.value
                          }))
                        }
                        type="number"
                        min={0}
                        max={99}
                        inputMode="numeric"
                        disabled={busy || Boolean(pendingScoreSave)}
                        aria-label={`${match.id} side A score`}
                        style={inputStyle}
                      />
                      <input
                        value={scores[scoreKey(match.id, "b")] ?? ""}
                        onChange={(event_) =>
                          setScores((current) => ({
                            ...current,
                            [scoreKey(match.id, "b")]: event_.target.value
                          }))
                        }
                        type="number"
                        min={0}
                        max={99}
                        inputMode="numeric"
                        disabled={busy || Boolean(pendingScoreSave)}
                        aria-label={`${match.id} side B score`}
                        style={inputStyle}
                      />
                    </>
                  ) : (
                    <strong>
                      {!scoredSession
                        ? "vs."
                        : match.status === "not_played" ? "Not played" : match.scoreA == null || match.scoreB == null
                          ? "—"
                          : `${match.scoreA}–${match.scoreB}`}
                    </strong>
                  )}
                  <strong style={{ textAlign: "right" }}>{sideLabel(match, "B", participants)}</strong>
                </div>
              </article>
            );
          })}
        </div>

        {canEditRound ? (
          <div style={{ marginTop: "1rem", display: "grid", gap: "0.75rem" }}>
            <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
              {scoredSession ? (
                <button type="button" onClick={() => void saveRound()} disabled={busy} style={primaryButton}>
                  {busy ? "Saving…" : pendingScoreSave ? "Retry score save" : "Save round scores"}
                </button>
              ) : (
                <button type="button" onClick={() => void markRoundPlayed()} disabled={busy || Boolean(pendingScoreSave)} style={primaryButton}>
                  {busy ? "Saving…" : "Mark round played"}
                </button>
              )}
              {round.stage !== "playoff" && event.eventFormat !== "moneyball_expectations" ? <><input
                value={skipReason}
                onChange={(event_) => setSkipReason(event_.target.value)}
                placeholder="Optional skip reason"
                style={{ ...inputStyle, maxWidth: 260 }}
              />
              <ConfirmAction
                triggerLabel="Skip round"
                title={`Skip Round ${roundNumber}?`}
                description={
                  anyDraftScore
                    ? "This skips this round and permanently discards the unsaved score entries shown below."
                    : "This skips this round without saving a result."
                }
                preview={
                  <div style={{ display: "grid", gap: "0.35rem" }}>
                    <p style={{ margin: 0 }}>
                      <strong>Unsaved score entries:</strong> {draftScoreCount} {draftScoreCount === 1 ? "entry" : "entries"}
                      {draftScoreCount ? " will be discarded" : ""}
                    </p>
                    <p style={{ margin: 0 }}>
                      <strong>Skip reason:</strong> {skipReason.trim() || "No reason provided"}
                    </p>
                  </div>
                }
                confirmLabel="Yes, skip round"
                confirmationText="SKIP ROUND"
                tone={anyDraftScore ? "danger" : "default"}
                disabled={busy || Boolean(pendingScoreSave)}
                busy={busy}
                onConfirm={skipRound}
                onAcknowledge={acknowledgeSkip}
              />
              </> : null}
            </div>
          </div>
        ) : null}

        {round.status === "saved" && scoredSession ? (
          <section style={{ marginTop: "1rem", borderTop: "1px solid #e2e8f0", paddingTop: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
              <h3 style={{ margin: 0 }}>Round {roundNumber} results</h3>
              {generatorKind === "round_robin" && scoredSession ? (
                <Link href={sessionStandingsPath(clubId, sessionKey)} style={secondaryButton}>View full standings</Link>
              ) : null}
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th align="left">Player</th>
                    <th>W</th>
                    <th>L</th>
                    <th>Diff</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row) => (
                    <tr key={row.participantId}>
                      <td>{row.name}</td>
                      <td align="center">{row.wins}</td>
                      <td align="center">{row.losses}</td>
                      <td align="center">{row.differential}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {round.status === "played" ? (
          <p style={{ marginTop: "1rem", padding: "0.7rem", background: "#dcfce7", borderRadius: "8px" }}>
            Round {roundNumber} was marked played.
          </p>
        ) : null}

        {round.status === "skipped" ? (
          <div style={{ marginTop: "1rem", padding: "0.7rem", background: "#fef3c7", borderRadius: "8px" }}>
            <p>This round was skipped{round.skipReason ? `: ${round.skipReason}` : "."}</p>
            {canReopenRound ? (
              <button type="button" onClick={() => void reopenRound()} disabled={busy || Boolean(pendingScoreSave)} style={primaryButton}>
                {busy ? "Opening…" : "Play this round"}
              </button>
            ) : null}
          </div>
        ) : null}

        {canManage && !editingScores && isCurrent && ["saved", "played", "skipped"].includes(round.status) && session.status === "active" ? (
            <button
              type="button"
              onClick={() => void advanceRound()}
              disabled={busy || Boolean(pendingScoreSave)}
              style={{ ...primaryButton, marginTop: "1rem" }}
            >
              {event.eventFormat === "moneyball_expectations" && event.rounds.every(row => row.status === "saved")
                ? "Finish Moneyball & view results"
                : generatorKind === "ladder" && roundNumber >= event.totalRounds
                ? "Finish session"
                : generatorKind === "ladder"
                  ? `Generate Round ${roundNumber + 1}`
                  : "Next scheduled round"}
            </button>
        ) : null}
        {canFinishSession && !editingScores && event.eventFormat !== "moneyball_expectations" ? (
          <div style={{ marginTop: "1rem" }}>
            <ConfirmAction
              triggerLabel="Finish session"
              title="Finish this session?"
              description={round.status === "active"
                ? "This unfinished round will not count. Saved scores and played rounds will be kept."
                : session.round_robin_hub_id
                  ? "Finish with the saved scores and played rounds."
                  : "Finish with the saved scores and played rounds. You can keep playing again until results are submitted."}
              confirmLabel="Finish session"
              confirmationText="FINISH SESSION"
              disabled={busy || Boolean(pendingScoreSave) || (round.status === "active" && anyDraftScore)}
              busy={busy}
              onConfirm={finishSession}
            />
            {round.status === "active" && anyDraftScore ? <p>Save this round’s scores before finishing the session.</p> : null}
          </div>
        ) : null}
      </article>

      {canManage && isCurrent && session.status === "active" && Boolean(editToken) && !event.playoff ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{event.eventFormat === "moneyball_expectations" ? "Substitute a player" : "Change players"}</h2>
          <p style={{ color: "#475569" }}>
            {event.eventFormat === "moneyball_expectations" ? "A substitute fills the same scheduled slot. Both the original player and substitute become ineligible to win. Saved games and everyone else’s expectations stay fixed." : generatorKind === "round_robin" || roundNumber < event.totalRounds
              ? "Changes apply from the next round. Current games, byes and scores stay in place."
              : "This is the final round. Current games, byes and scores stay in place."}
            {event.eventFormat === "moneyball_expectations" ? "" : generatorKind === "round_robin" && ["singles", "doubles"].includes(event.playFormat)
              ? " Add late arrivals here; they can join players on a bye for a game on a spare court below."
              : " Added players join from the next round."}
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "0.75rem"
            }}
          >
            <label>
              Change
              <br />
              <select
                value={rosterAction}
                onChange={(event_) =>
                  setRosterAction(event_.target.value as RosterAction)
                }
                style={inputStyle}
              >
                {event.eventFormat !== "moneyball_expectations" ? <><option value="add">Add player</option>
                <option value="remove">Remove player</option></> : null}
                <option value="substitute">Substitute player</option>
                {event.eventFormat !== "moneyball_expectations" ? <><option value="swap">Swap players</option>
                <option value="reorder">Reorder roster</option></> : null}
              </select>
            </label>

            {rosterAction === "swap" ? (
              <>
                <label>
                  First player
                  <br />
                  <SearchablePlayerSelect aria-label="First player"
                    value={firstSwapParticipant}
                    onValueChange={playerValue => {
                      const nextParticipant = playerValue;
                      setFirstSwapParticipant(nextParticipant);
                      if (nextParticipant === secondSwapParticipant) {
                        setSecondSwapParticipant("");
                      }
                    }}
                    style={inputStyle}
                  >
                    <option value="">Select player</option>
                    {rosterOrder.map((id, index) => (
                      <option key={id} value={id}>
                        {index + 1}. {participants.get(id)?.name || id}
                      </option>
                    ))}
                  </SearchablePlayerSelect>
                </label>
                <label>
                  Second player
                  <br />
                  <SearchablePlayerSelect aria-label="Second player"
                    value={secondSwapParticipant}
                    onValueChange={playerValue => setSecondSwapParticipant(playerValue)}
                    style={inputStyle}
                  >
                    <option value="">Select player</option>
                    {rosterOrder.map((id, index) =>
                      id === firstSwapParticipant ? null : (
                        <option key={id} value={id}>
                          {index + 1}. {participants.get(id)?.name || id}
                        </option>
                      )
                    )}
                  </SearchablePlayerSelect>
                </label>
              </>
            ) : null}

            {["remove", "substitute"].includes(rosterAction) ? (
              <label>
                Current player
                <br />
                <SearchablePlayerSelect aria-label="Current player"
                  value={selectedParticipant}
                  onValueChange={playerValue => setSelectedParticipant(playerValue)}
                  style={inputStyle}
                >
                  <option value="">Select player</option>
                  {[...event.participants]
                    .sort(
                      (left, right) =>
                        Number(left.roster_order || 0) - Number(right.roster_order || 0)
                    )
                    .map((participant) => (
                      <option key={participant.id} value={participant.id}>
                        {participant.name}
                      </option>
                    ))}
                </SearchablePlayerSelect>
              </label>
            ) : null}

            {["add", "substitute"].includes(rosterAction) ? (
              <>
                <label>
                  New player name
                  <br />
                  <input
                    value={newPlayerName}
                    onChange={(event_) => setNewPlayerName(event_.target.value)}
                    style={inputStyle}
                  />
                </label>
                <label>
                  Club player ID (optional)
                  <br />
                  <input
                    value={newPlayerId}
                    onChange={(event_) => setNewPlayerId(event_.target.value)}
                    inputMode="numeric"
                    style={inputStyle}
                  />
                </label>
              </>
            ) : null}

            {rosterAction === "substitute" ? (
              <label>
                Apply substitution to
                <br />
                <select
                  value={substituteScope}
                  onChange={(event_) =>
                    setSubstituteScope(event_.target.value as "round" | "rest")
                  }
                  style={inputStyle}
                >
                  <option value="rest">Rest of session</option>
                  <option value="round">{event.eventFormat === "moneyball_expectations" ? "Current unscored round only" : "Next round only"}</option>
                </select>
              </label>
            ) : null}
          </div>

          {rosterAction === "reorder" ? (
            <div style={{ display: "grid", gap: "0.45rem", marginTop: "0.75rem" }}>
              {rosterOrder.map((id, index) => (
                <div
                  key={id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "2rem minmax(0, 1fr) auto",
                    gap: "0.5rem",
                    alignItems: "center"
                  }}
                >
                  <strong>{index + 1}</strong>
                  <span>{participants.get(id)?.name || id}</span>
                  <div style={{ display: "flex", gap: "0.3rem" }}>
                    <button
                      type="button"
                      onClick={() => moveRoster(index, -1)}
                      disabled={index === 0}
                      style={secondaryButton}
                    >
                      ↑
                    </button>
                    <button
                      type="button"
                      onClick={() => moveRoster(index, 1)}
                      disabled={index === rosterOrder.length - 1}
                      style={secondaryButton}
                    >
                      ↓
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <button
            type="button"
            onClick={() => void saveRosterChange()}
            disabled={
              busy ||
              (event.eventFormat === "moneyball_expectations" && (editingScores || (canEditRound && anyDraftScore))) ||
              (rosterAction === "remove" && !selectedParticipant) ||
              (rosterAction === "substitute" &&
                (!selectedParticipant || !newPlayerName.trim())) ||
              (rosterAction === "swap" &&
                (!firstSwapParticipant ||
                  !secondSwapParticipant ||
                  firstSwapParticipant === secondSwapParticipant)) ||
              (rosterAction === "add" && !newPlayerName.trim())
            }
            style={{ ...primaryButton, marginTop: "0.8rem" }}
          >
            {rosterAction === "add" ? (generatorKind === "round_robin" || roundNumber < event.totalRounds ? "Add player for next round" : "Add late arrival") : event.eventFormat === "moneyball_expectations" ? "Apply substitution · both players ineligible" : "Apply change for next round"}
          </button>
          {event.eventFormat !== "moneyball_expectations" ? <GeneratorLateArrivals
            participants={event.participants}
            roundNumber={roundNumber}
            totalRounds={event.totalRounds}
            generatorKind={generatorKind}
            playFormat={event.playFormat}
            roundStatus={round.status}
            courtCount={event.courtCount}
            usedCourts={matches.map(match => Number(match.court || 0))}
            byeParticipantIds={round.byeParticipantIds}
            busy={busy}
            queueOnly={Boolean(session.round_robin_hub_id)}
            onStart={startArrivalGame}
          /> : null}
        </article>
      ) : null}

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Ratings</h2>
        <p style={{ color: "#475569" }}>
          {session.rating_mode === "rated" ? "Ratings update only after admin approval." : "These games won’t affect ratings."} Share this page with players or download the results below.
        </p>
        {!shared ? <a href={`/api/clubs/${clubId}/play-generators/sessions/${sessionKey}/export?format=csv`}>
          Download session CSV
        </a> : null}
      </article>

      {message ? (
        <p
          role="status"
          aria-live="polite"
          style={{ color: /couldn|unable|error|must|requires|changed|not found/i.test(message) ? "#b91c1c" : "#166534" }}
        >
          {message}
        </p>
      ) : null}
    </div>
  );
}

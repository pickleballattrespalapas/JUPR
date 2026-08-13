"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import { useAdminSession } from "@/lib/useAdminSession";

type GeneratorKind = "round_robin" | "ladder";
type ScoringMode = "scored" | "unscored";
type PlayFormat = "singles" | "doubles" | "doubles_singles";

type Participant = {
  id: string;
  name: string;
  player_id?: number | null;
  roster_order?: number;
  active_from_round?: number;
  inactive_from_round?: number | null;
  inactive_rounds?: number[];
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
  name: string;
  generatorKind: GeneratorKind;
  playFormat: PlayFormat;
  scoringMode?: ScoringMode;
  status: string;
  currentRoundNumber: number;
  totalRounds: number;
  participants: Participant[];
  rounds: RoundRow[];
  rosterRevisions?: Array<Record<string, unknown>>;
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
  event: GeneratorEvent;
  official_publish?: {
    published_match_ids?: string[];
    published_at?: string | null;
  };
};

type DetailResponse = {
  ok: boolean;
  session: GeneratorSession;
};

type MutationResponse = {
  ok: boolean;
  session?: GeneratorSession;
  published_count?: number;
};

type SkipRoundRequest = {
  skipBody: {
    reason: string;
    expected_version: string;
    idempotency_key: string;
  };
  advanceIdempotencyKey: string;
};

type PublishMatchesRequest = {
  match_date: string | null;
  confirmation_text: string;
  expected_version: string;
  idempotency_key: string;
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

type Props = {
  apiBase: string | null;
  clubId: string;
  generatorKind: GeneratorKind;
  sessionKey: string;
  roundNumber: number;
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

function apiErrorMessage(detail: unknown, status: number): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    const message = String((detail as { message?: unknown }).message || "").trim();
    if (message) return message;
  }
  return `API error (${status})`;
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

function roundPath(kind: GeneratorKind, sessionKey: string, round: number): string {
  return `/admin/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;
}

function standingsPath(sessionKey: string): string {
  return `/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/standings`;
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
    if (match.scoreA == null || match.scoreB == null) continue;
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
  roundNumber
}: Props) {
  const router = useRouter();
  const { accessToken } = useAdminSession();
  const [session, setSession] = useState<GeneratorSession | null>(null);
  const [scores, setScores] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [skipReason, setSkipReason] = useState("");
  const [rosterAction, setRosterAction] = useState<"add" | "remove" | "substitute" | "reorder">("add");
  const [selectedParticipant, setSelectedParticipant] = useState("");
  const [newPlayerName, setNewPlayerName] = useState("");
  const [newPlayerId, setNewPlayerId] = useState("");
  const [substituteScope, setSubstituteScope] = useState<"round" | "rest">("rest");
  const [rosterOrder, setRosterOrder] = useState<string[]>([]);
  const [publishDate, setPublishDate] = useState("");
  const skipDestinationRef = useRef<number | "completed" | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new ApiRequestError("Missing API base URL.", 400, "local_configuration");
    if (!accessToken) throw new ApiRequestError("Sign in before managing this session.", 401, "local_authentication");
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
      const detail = payload?.detail;
      throw new ApiRequestError(apiErrorMessage(detail, response.status), response.status, detail);
    }
    return payload as T;
  }

  function applySession(next: GeneratorSession): void {
    setSession(next);
    const nextScores: Record<string, string> = {};
    const requestedRound = next.event.rounds.find((row) => row.number === roundNumber);
    for (const match of flattenMatches(requestedRound || null)) {
      nextScores[scoreKey(match.id, "a")] = match.scoreA == null ? "" : String(match.scoreA);
      nextScores[scoreKey(match.id, "b")] = match.scoreB == null ? "" : String(match.scoreB);
    }
    setScores(nextScores);
    const ordered = [...next.event.participants]
      .sort(
        (left, right) =>
          Number(left.roster_order || 0) - Number(right.roster_order || 0)
      )
      .map((row) => row.id);
    setRosterOrder(ordered);
  }

  async function loadSession(): Promise<void> {
    if (!accessToken) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<DetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}`
      );
      if (payload.session.generator_kind !== generatorKind) {
        throw new Error("This session belongs to the other generator module.");
      }
      applySession(payload.session);
      const hasRequested = payload.session.event.rounds.some((row) => row.number === roundNumber);
      if (!hasRequested) {
        router.replace(
          roundPath(
            generatorKind,
            sessionKey,
            payload.session.current_round_number || 1
          )
        );
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load the session.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void loadSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, sessionKey, roundNumber]);

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
  const scoringMode: ScoringMode = session?.scoring_mode || event?.scoringMode || "scored";
  const scoredSession = scoringMode === "scored";
  const canEditRound =
    Boolean(session) &&
    session?.status === "active" &&
    isCurrent &&
    round?.status === "active";
  const draftScoreCount = scoredSession
    ? Object.values(scores).filter((value) => value !== "").length
    : 0;
  const anyDraftScore = draftScoreCount > 0;
  const publishedMatchIds = new Set(
    (session?.official_publish?.published_match_ids || []).map(String)
  );
  const unpublishedSavedMatchCount = (event?.rounds || []).reduce(
    (count, row) =>
      count +
      (row.status === "saved"
        ? flattenMatches(row).filter(
            (match) =>
              match.scoreA != null &&
              match.scoreB != null &&
              !publishedMatchIds.has(String(match.id))
          ).length
        : 0),
    0
  );
  const results = round ? roundStandings(round, participants) : [];
  const byeNames = (round?.byeParticipantIds || [])
    .map((id) => participants.get(id)?.name || id)
    .join(", ");

  async function runMutation(path: string, body: Record<string, unknown>): Promise<GeneratorSession> {
    const payload = await requestJson<MutationResponse>(path, {
      method: "POST",
      body: JSON.stringify({
        ...body,
        expected_version: session?.version || "",
        idempotency_key: operationKey(String(body.action || path.split("/").pop() || "mutation"))
      })
    });
    if (!payload.session) throw new Error("The operation completed without a refreshed session.");
    applySession(payload.session);
    return payload.session;
  }

  async function saveRound(): Promise<void> {
    if (!session || !round) return;
    setBusy(true);
    setMessage(null);
    try {
      const scoreRows = matches.map((match) => {
        const left = scores[scoreKey(match.id, "a")] ?? "";
        const right = scores[scoreKey(match.id, "b")] ?? "";
        return {
          match_id: match.id,
          score_a: left === "" ? null : Number(left),
          score_b: right === "" ? null : Number(right)
        };
      });
      const payload = await requestJson<MutationResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/scores`,
        {
          method: "PATCH",
          body: JSON.stringify({
            scores: scoreRows,
            expected_version: session.version,
            idempotency_key: operationKey("scores")
          })
        }
      );
      if (!payload.session) throw new Error("Scores saved without a refreshed session.");
      applySession(payload.session);
      setMessage(`Round ${roundNumber} scores saved.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save the round.");
    } finally {
      setBusy(false);
    }
  }

  async function markRoundPlayed(): Promise<void> {
    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;
    setBusy(true);
    setMessage(null);
    try {
      const playedPayload = await requestJson<MutationResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/played`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_version: session.version,
            idempotency_key: operationKey("played")
          })
        }
      );
      if (!playedPayload.session) {
        throw new Error("Round marked played without a refreshed session.");
      }
      applySession(playedPayload.session);
      if (playedPayload.session.status === "completed") {
        setMessage("Session completed.");
        router.refresh();
        return;
      }
      const nextRound = playedPayload.session.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to mark the round played.");
    } finally {
      setBusy(false);
    }
  }


  async function executeSkipRound(request: SkipRoundRequest): Promise<ActionCompletion> {
    setBusy(true);
    setMessage(null);
    let skipCommitted = false;
    try {
      const payload = await requestJson<MutationResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/skip`,
        {
          method: "POST",
          body: JSON.stringify(request.skipBody)
        }
      );
      if (!payload.session) throw new Error("Round skipped without a refreshed session.");
      skipCommitted = true;
      applySession(payload.session);
      if (generatorKind === "round_robin" && !scoredSession) {
        const advancedPayload = await requestJson<MutationResponse>(
          `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
            sessionKey
          )}/advance`,
          {
            method: "POST",
            body: JSON.stringify({
              expected_version: payload.session.version,
              idempotency_key: request.advanceIdempotencyKey
            })
          }
        );
        if (!advancedPayload.session) throw new Error("Skipped round advanced without a refreshed session.");
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
      const errorMessage = error instanceof Error ? error.message : "Unable to skip the round.";
      if (skipCommitted || isUncertainRequestError(error)) {
        const recoveryMessage = skipCommitted
          ? `${errorMessage} The round skip completed as ${request.skipBody.idempotency_key}, but automatic advance ${request.advanceIdempotencyKey} was not confirmed. Both exact operation keys are retained; resume this exact skip-and-advance flow before starting another action.`
          : `${errorMessage} The exact skip request is retained as ${request.skipBody.idempotency_key}; retry it here before starting another action.`;
        setMessage(recoveryMessage);
        return actionUncertain(
          skipCommitted ? "Round advance needs verification" : "Round skip needs verification",
          recoveryMessage,
          request.skipBody.idempotency_key,
          skipCommitted ? "Resume exact skip and advance" : "Retry exact skip request",
          () => executeSkipRound(request)
        );
      }
      setMessage(errorMessage);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  function skipRound(): Promise<ActionCompletion> {
    if (!session || !round) throw new Error("Reload the session before skipping this round.");
    skipDestinationRef.current = null;
    return executeSkipRound({
      skipBody: {
        reason: skipReason,
        expected_version: session.version,
        idempotency_key: operationKey("skip")
      },
      advanceIdempotencyKey: operationKey("advance-after-skip")
    });
  }

  function acknowledgeSkip(): void {
    const destination = skipDestinationRef.current;
    skipDestinationRef.current = null;
    if (typeof destination === "number") {
      router.push(roundPath(generatorKind, sessionKey, destination));
    }
    if (destination !== null) router.refresh();
  }

  async function advanceRound(): Promise<void> {
    if (!session) return;
    setBusy(true);
    setMessage(null);
    try {
      const next = await runMutation(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/advance`,
        {}
      );
      if (next.status === "completed") {
        setMessage("Session completed. You can review or publish the saved matches.");
        router.refresh();
        return;
      }
      const nextRound = next.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to advance the round.");
    } finally {
      setBusy(false);
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
    if (!session) return;
    if (anyDraftScore) {
      setMessage("Save or clear the current score entries before changing the roster.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const body: Record<string, unknown> = { action: rosterAction };
      if (rosterAction === "reorder") {
        body.roster_order = rosterOrder;
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
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/roster`,
        body
      );
      setSelectedParticipant("");
      setNewPlayerName("");
      setNewPlayerId("");
      setMessage(
        rosterAction === "substitute"
          ? "Substitution saved. Completed rounds remain unchanged."
          : "Roster updated. Future matchups were regenerated when applicable."
      );
      const current = next.current_round_number || roundNumber;
      if (current !== roundNumber) {
        router.push(roundPath(generatorKind, sessionKey, current));
      } else {
        router.refresh();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update the roster.");
    } finally {
      setBusy(false);
    }
  }

  async function executePublishMatches(request: PublishMatchesRequest): Promise<ActionCompletion> {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<MutationResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/publish`,
        {
          method: "POST",
          body: JSON.stringify(request)
        }
      );
      if (!payload.ok || !payload.session) {
        throw new Error("Official matches published without a refreshed session.");
      }
      applySession(payload.session);
      const publishedCount = payload.published_count ?? 0;
      const successMessage = `Published ${publishedCount} official rated ${publishedCount === 1 ? "match" : "matches"}.`;
      setMessage(successMessage);
      return actionSuccess("Official matches published", successMessage);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to publish matches.";
      if (isUncertainRequestError(error)) {
        const recoveryMessage = `${errorMessage} The exact publish request is retained as ${request.idempotency_key}; retry it here before publishing again.`;
        setMessage(recoveryMessage);
        return actionUncertain(
          "Official publish needs verification",
          recoveryMessage,
          request.idempotency_key,
          "Retry exact publish request",
          () => executePublishMatches(request)
        );
      }
      setMessage(errorMessage);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  function publishMatches(confirmationText: string): Promise<ActionCompletion> {
    if (!session) throw new Error("Reload the session before publishing official matches.");
    return executePublishMatches({
      match_date: publishDate || null,
      confirmation_text: confirmationText,
      expected_version: session.version,
      idempotency_key: operationKey("publish")
    });
  }

  if (!session || !event || !round) {
    return (
      <article style={cardStyle}>
        <h1>{generatorTitle(generatorKind)}</h1>
        <p>{busy ? "Loading session…" : message || "Session unavailable."}</p>
        <Link href={`/admin/${generatorSlug(generatorKind)}`}>Back to generator</Link>
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
          <Link href={`/admin/${generatorSlug(generatorKind)}`}>
            ← {generatorTitle(generatorKind)}
          </Link>
        </p>
        <h1 style={{ margin: "0 0 0.4rem" }}>{session.title}</h1>
        <p style={{ margin: 0, color: "#475569" }}>
          {playFormatLabel(session.play_format)} · Round {roundNumber} of{" "}
          {event.totalRounds} · {scoredSession ? "Scored" : "Unscored"} · {round.status}
        </p>
      </article>

      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Session complete</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>All scheduled rounds are complete. Review the saved session history below.</p>
        </article>
      ) : null}

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>Round {roundNumber}</h2>
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
              <Link href={standingsPath(sessionKey)} style={secondaryButton}>
                Standings
              </Link>
            ) : null}
            {previousRound ? (
              <Link href={roundPath(generatorKind, sessionKey, previousRound)} style={secondaryButton}>
                Previous round
              </Link>
            ) : null}
            {nextExisting && nextExisting <= Number(event.currentRoundNumber || 1) ? (
              <Link href={roundPath(generatorKind, sessionKey, nextExisting)} style={secondaryButton}>
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

        <div style={{ display: "grid", gap: "0.75rem" }}>
          {matches.map((match) => {
            const editable = canEditRound && scoredSession;
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
                  {matchFormatLabel(match, event.playFormat)} · Court {match.court || "—"}
                  {match.miniRound ? ` · Game ${match.miniRound}` : ""}
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
                        aria-label={`${match.id} side B score`}
                        style={inputStyle}
                      />
                    </>
                  ) : (
                    <strong>
                      {!scoredSession
                        ? "vs."
                        : match.scoreA == null || match.scoreB == null
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
                  {busy ? "Saving…" : "Save round scores"}
                </button>
              ) : (
                <button type="button" onClick={() => void markRoundPlayed()} disabled={busy} style={primaryButton}>
                  {busy ? "Saving…" : "Round Played"}
                </button>
              )}
              <input
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
                    ? "This skips the current round and permanently discards the unsaved score entries shown below."
                    : "This skips the current round without saving a result."
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
                disabled={busy}
                busy={busy}
                onConfirm={skipRound}
                onAcknowledge={acknowledgeSkip}
              />
            </div>
          </div>
        ) : null}

        {round.status === "saved" && scoredSession ? (
          <section style={{ marginTop: "1rem", borderTop: "1px solid #e2e8f0", paddingTop: "1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
              <h3 style={{ margin: 0 }}>Round {roundNumber} results</h3>
              {generatorKind === "round_robin" && scoredSession ? (
                <Link href={standingsPath(sessionKey)} style={secondaryButton}>View full standings</Link>
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
          <p style={{ marginTop: "1rem", padding: "0.7rem", background: "#fef3c7", borderRadius: "8px" }}>
            This round was skipped{round.skipReason ? `: ${round.skipReason}` : "."}
          </p>
        ) : null}

        {isCurrent && ["saved", "played", "skipped"].includes(round.status) && session.status === "active" ? (
          generatorKind === "round_robin" && scoredSession ? (
            <Link href={standingsPath(sessionKey)} style={{ ...primaryButton, display: "inline-flex", marginTop: "1rem", textDecoration: "none" }}>
              View standings and continue
            </Link>
          ) : (
            <button
              type="button"
              onClick={() => void advanceRound()}
              disabled={busy}
              style={{ ...primaryButton, marginTop: "1rem" }}
            >
              {roundNumber >= event.totalRounds
                ? "Finish session"
                : generatorKind === "ladder"
                  ? `Generate Round ${roundNumber + 1}`
                  : `Continue to Round ${roundNumber + 1}`}
            </button>
          )
        ) : null}
      </article>

      {isCurrent && session.status === "active" ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Adaptive roster</h2>
          <p style={{ color: "#475569" }}>
            Completed rounds never change. If this round has no saved scores, roster changes can regenerate it.
            Otherwise, changes take effect in the next round.
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
                  setRosterAction(
                    event_.target.value as "add" | "remove" | "substitute" | "reorder"
                  )
                }
                style={inputStyle}
              >
                <option value="add">Add player</option>
                <option value="remove">Remove player</option>
                <option value="substitute">Substitute player</option>
                <option value="reorder">Reorder roster</option>
              </select>
            </label>

            {["remove", "substitute"].includes(rosterAction) ? (
              <label>
                Current player
                <br />
                <select
                  value={selectedParticipant}
                  onChange={(event_) => setSelectedParticipant(event_.target.value)}
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
                </select>
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
                  Official player ID optional
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
                Substitute scope
                <br />
                <select
                  value={substituteScope}
                  onChange={(event_) =>
                    setSubstituteScope(event_.target.value as "round" | "rest")
                  }
                  style={inputStyle}
                >
                  <option value="rest">Rest of session</option>
                  <option value="round">One round only</option>
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
              (rosterAction === "remove" && !selectedParticipant) ||
              (rosterAction === "substitute" &&
                (!selectedParticipant || !newPlayerName.trim())) ||
              (rosterAction === "add" && !newPlayerName.trim())
            }
            style={{ ...primaryButton, marginTop: "0.8rem" }}
          >
            Apply roster change
          </button>
        </article>
      ) : null}

      {scoredSession ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Official results</h2>
          <p style={{ color: "#475569" }}>
            Publication requires an official player ID for every player in each saved match. Singles games publish
            to singles ratings, and doubles games publish to doubles ratings.
          </p>
          <div style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", alignItems: "end" }}>
            <label>
              Match date optional
              <br />
              <input
                value={publishDate}
                onChange={(event_) => setPublishDate(event_.target.value)}
                placeholder="Defaults to publish time"
                style={inputStyle}
              />
            </label>
            <ConfirmAction
              triggerLabel="Publish official matches"
              title={`Publish ${unpublishedSavedMatchCount} official rated ${unpublishedSavedMatchCount === 1 ? "match" : "matches"}?`}
              description="This publishes every unpublished saved result to official rating history. Review the exact count and match date before continuing."
              preview={
                <div style={{ display: "grid", gap: "0.35rem" }}>
                  <p style={{ margin: 0 }}>
                    <strong>Unpublished saved results:</strong> {unpublishedSavedMatchCount} {unpublishedSavedMatchCount === 1 ? "match" : "matches"}
                  </p>
                  <p style={{ margin: 0 }}>
                    <strong>Match date:</strong> {publishDate || "Publish time (default)"}
                  </p>
                </div>
              }
              confirmLabel="Yes, publish official matches"
              confirmationText="PUBLISH MATCHES"
              tone="danger"
              disabled={busy || unpublishedSavedMatchCount === 0}
              disabledReason={unpublishedSavedMatchCount === 0 ? "No unpublished saved matches are ready." : undefined}
              busy={busy}
              onConfirm={publishMatches}
            />
          </div>
          <p style={{ color: "#64748b" }}>
            Published: {session.official_publish?.published_match_ids?.length || 0} match(es)
          </p>
        </article>
      ) : null}

      {message ? (
        <p
          role="status"
          aria-live="polite"
          style={{ color: /unable|error|must|requires|changed|not found/i.test(message) ? "#b91c1c" : "#166534" }}
        >
          {message}
        </p>
      ) : null}
    </div>
  );
}

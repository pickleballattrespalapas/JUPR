"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ConfirmAction } from "@/components/ConfirmAction";
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
    expected_version: number;
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
  const [editToken, setEditToken] = useState("");
  const [session, setSession] = useState<GeneratorSession | null>(null);
  const [scores, setScores] = useState<Record<string, string>>({});
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

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) {
      throw new ApiRequestError(
        "This play tool is temporarily unavailable. Please try again.",
        503,
        "local_configuration"
      );
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
      const detail = payload?.detail;
      throw new ApiRequestError(apiErrorMessage(response.status, detail), response.status, detail);
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
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<DetailResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}`
      );
      if (payload.session.generator_kind !== generatorKind) {
        throw new UserFacingError("This link opens in the other play generator.");
      }
      applySession(payload.session);
      const hasRequested = payload.session.event.rounds.some((row) => row.number === roundNumber);
      if (!hasRequested) {
        router.replace(
          roundPath(
            generatorKind,
            clubId,
            sessionKey,
            payload.session.current_round_number || 1
          )
        );
      }
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t load this session. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const storageKey = `public-generator-edit:${clubId}:${sessionKey}`;
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";
    if (discovered) {
      sessionStorage.setItem(storageKey, discovered);
      setEditToken(discovered);
    }
    if (hash.has("edit")) {
      window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);
    }
    void loadSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clubId, sessionKey, roundNumber]);

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
    Boolean(editToken) &&
    session?.status === "active" &&
    isCurrent &&
    round?.status === "active";
  const draftScoreCount = scoredSession
    ? Object.values(scores).filter((value) => value !== "").length
    : 0;
  const anyDraftScore = draftScoreCount > 0;
  const results = round ? roundStandings(round, participants) : [];
  const byeNames = (round?.byeParticipantIds || [])
    .map((id) => participants.get(id)?.name || id)
    .join(", ");

  async function runMutation(path: string, body: Record<string, unknown>): Promise<GeneratorSession> {
    const payload = await requestJson<MutationResponse>(path, {
      method: "POST",
      body: JSON.stringify({
        ...body,
        edit_token: editToken,
        expected_version: Number(session?.version || 1),
        idempotency_key: operationKey(String(body.action || path.split("/").pop() || "mutation"))
      })
    });
    if (!payload.session) throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
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
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/scores`,
        {
          method: "PATCH",
          body: JSON.stringify({
            scores: scoreRows,
            edit_token: editToken,
            expected_version: Number(session.version),
            idempotency_key: operationKey("scores")
          })
        }
      );
      if (!payload.session) throw new UserFacingError("We couldn’t load the latest session. Refresh the page and check whether your change was saved.");
      applySession(payload.session);
      setMessage(`Round ${roundNumber} scores saved.`);
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t save the scores. Please try again."));
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
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/played`,
        {
          method: "POST",
          body: JSON.stringify({
            edit_token: editToken,
            expected_version: Number(session.version),
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
      router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t update the round. Please try again."));
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
      if (generatorKind === "round_robin" && !scoredSession) {
        const advancedPayload = await requestJson<MutationResponse>(
          `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
            sessionKey
          )}/advance`,
          {
            method: "POST",
            body: JSON.stringify({
              edit_token: request.skipBody.edit_token,
              expected_version: Number(payload.session.version),
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
      setBusy(false);
    }
  }

  function skipRound(): Promise<ActionCompletion> {
    if (!session || !round) throw new UserFacingError("Reload the session before skipping this round.");
    skipDestinationRef.current = null;
    return executeSkipRound({
      skipBody: {
        reason: skipReason,
        edit_token: editToken,
        expected_version: Number(session.version),
        idempotency_key: operationKey("skip")
      },
      advanceIdempotencyKey: operationKey("advance-after-skip")
    });
  }

  function acknowledgeSkip(): void {
    const destination = skipDestinationRef.current;
    skipDestinationRef.current = null;
    if (typeof destination === "number") {
      router.push(roundPath(generatorKind, clubId, sessionKey, destination));
    }
    if (destination !== null) router.refresh();
  }

  async function advanceRound(): Promise<void> {
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
        router.refresh();
        return;
      }
      const nextRound = next.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t continue to the next round. Please try again."));
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
        body
      );
      setSelectedParticipant("");
      setFirstSwapParticipant("");
      setSecondSwapParticipant("");
      setNewPlayerName("");
      setNewPlayerId("");
      setMessage(
        rosterAction === "substitute"
          ? "Player changed. Completed rounds stay the same."
          : rosterAction === "swap"
            ? "Players swapped. Completed rounds are unchanged, and upcoming matchups have been updated."
          : "Roster updated. Upcoming matchups have been adjusted."
      );
      const current = next.current_round_number || roundNumber;
      if (current !== roundNumber) {
        router.push(roundPath(generatorKind, clubId, sessionKey, current));
      } else {
        router.refresh();
      }
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t update the players. Please try again."));
    } finally {
      setBusy(false);
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
          {playFormatLabel(session.play_format)} · Round {roundNumber} of{" "}
          {event.totalRounds} · {scoredSession ? "Scores on" : "Scores off"} · {roundStatusLabel(round.status)} · Won’t affect ratings
        </p>
        {!editToken ? <p style={{ color: "#64748b" }}>Only the organizer can enter scores or change players.</p> : null}
      </article>

      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Session complete</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>All scheduled rounds are complete. These games won’t affect ratings.</p>
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
              <Link href={standingsPath(clubId, sessionKey)} style={secondaryButton}>
                Standings
              </Link>
            ) : null}
            {previousRound ? (
              <Link href={roundPath(generatorKind, clubId, sessionKey, previousRound)} style={secondaryButton}>
                Previous round
              </Link>
            ) : null}
            {nextExisting && nextExisting <= Number(event.currentRoundNumber || 1) ? (
              <Link href={roundPath(generatorKind, clubId, sessionKey, nextExisting)} style={secondaryButton}>
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
                  {busy ? "Saving…" : "Mark round played"}
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
                <Link href={standingsPath(clubId, sessionKey)} style={secondaryButton}>View full standings</Link>
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
            <Link href={standingsPath(clubId, sessionKey)} style={{ ...primaryButton, display: "inline-flex", marginTop: "1rem", textDecoration: "none" }}>
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

      {isCurrent && session.status === "active" && Boolean(editToken) ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Change players</h2>
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
                  setRosterAction(event_.target.value as RosterAction)
                }
                style={inputStyle}
              >
                <option value="add">Add player</option>
                <option value="remove">Remove player</option>
                <option value="substitute">Substitute player</option>
                <option value="swap">Swap players</option>
                <option value="reorder">Reorder roster</option>
              </select>
            </label>

            {rosterAction === "swap" ? (
              <>
                <label>
                  First player
                  <br />
                  <select
                    value={firstSwapParticipant}
                    onChange={(event_) => {
                      const nextParticipant = event_.target.value;
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
                  </select>
                </label>
                <label>
                  Second player
                  <br />
                  <select
                    value={secondSwapParticipant}
                    onChange={(event_) => setSecondSwapParticipant(event_.target.value)}
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
                  </select>
                </label>
              </>
            ) : null}

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
              (rosterAction === "swap" &&
                (!firstSwapParticipant ||
                  !secondSwapParticipant ||
                  firstSwapParticipant === secondSwapParticipant)) ||
              (rosterAction === "add" && !newPlayerName.trim())
            }
            style={{ ...primaryButton, marginTop: "0.8rem" }}
          >
            {rosterAction === "swap" ? "Swap player positions" : "Apply roster change"}
          </button>
        </article>
      ) : null}

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Ratings</h2>
        <p style={{ color: "#475569" }}>
          These games won’t affect ratings. Share this page with players or download the results below.
        </p>
        <a href={`/api/clubs/${clubId}/play-generators/sessions/${sessionKey}/export?format=csv`}>
          Download session CSV
        </a>
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

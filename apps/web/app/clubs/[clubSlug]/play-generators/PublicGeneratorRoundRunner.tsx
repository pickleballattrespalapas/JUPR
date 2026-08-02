"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

type GeneratorKind = "round_robin" | "ladder";

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
};

type RoundRow = {
  number: number;
  status: "preview" | "active" | "saved" | "skipped" | string;
  matches?: MatchRow[];
  courts?: Array<{
    courtNumber: number;
    participantIds?: string[];
    matches?: MatchRow[];
  }>;
  byeParticipantIds?: string[];
  warnings?: string[];
  skipReason?: string | null;
};

type GeneratorEvent = {
  name: string;
  generatorKind: GeneratorKind;
  playFormat: "singles" | "doubles";
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
  play_format: "singles" | "doubles";
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
  const [rosterAction, setRosterAction] = useState<"add" | "remove" | "substitute" | "reorder">("add");
  const [selectedParticipant, setSelectedParticipant] = useState("");
  const [newPlayerName, setNewPlayerName] = useState("");
  const [newPlayerId, setNewPlayerId] = useState("");
  const [substituteScope, setSubstituteScope] = useState<"round" | "rest">("rest");
  const [rosterOrder, setRosterOrder] = useState<string[]>([]);
  const [publishDate, setPublishDate] = useState("");

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing API base URL.");
    const headers = new Headers(options?.headers);
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
        throw new Error("This session belongs to the other generator module.");
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
      setMessage(error instanceof Error ? error.message : "Unable to load the session.");
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
  const canEditRound =
    Boolean(session) &&
    Boolean(editToken) &&
    session?.status === "active" &&
    isCurrent &&
    round?.status === "active";
  const anyDraftScore = Object.values(scores).some((value) => value !== "");
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
      if (!payload.session) throw new Error("Scores saved without a refreshed session.");
      applySession(payload.session);
      setMessage(`Round ${roundNumber} scores saved.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save the round.");
    } finally {
      setBusy(false);
    }
  }

  async function skipRound(): Promise<void> {
    if (!session || !round) return;
    if (anyDraftScore && !window.confirm("Discard the unsaved score entries and skip this round?")) {
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<MutationResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/skip`,
        {
          method: "POST",
          body: JSON.stringify({
            reason: skipReason,
            edit_token: editToken,
            expected_version: Number(session.version),
            idempotency_key: operationKey("skip")
          })
        }
      );
      if (!payload.session) throw new Error("Round skipped without a refreshed session.");
      applySession(payload.session);
      setMessage(`Round ${roundNumber} skipped.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to skip the round.");
    } finally {
      setBusy(false);
    }
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
        setMessage("Session completed. You can review or publish the saved matches.");
        router.refresh();
        return;
      }
      const nextRound = next.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));
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
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
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
        router.push(roundPath(generatorKind, clubId, sessionKey, current));
      } else {
        router.refresh();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update the roster.");
    } finally {
      setBusy(false);
    }
  }

  async function publishMatches(): Promise<void> {
    if (!session) return;
    if (!window.confirm("Publish all unpublished saved matches as official rated matches?")) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<MutationResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/publish`,
        {
          method: "POST",
          body: JSON.stringify({
            match_date: publishDate || null,
            expected_version: session.version,
            idempotency_key: operationKey("publish")
          })
        }
      );
      if (payload.session) applySession(payload.session);
      setMessage(`Published ${payload.published_count || 0} official match(es).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to publish matches.");
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
          {session.play_format === "singles" ? "Singles" : "Doubles"} · Round {roundNumber} of{" "}
          {event.totalRounds} · {round.status} · Unrated public session
        </p>
        {!editToken ? <p style={{ color: "#64748b" }}>View-only link. The private organizer link enables scores and roster changes.</p> : null}
      </article>

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>Round {roundNumber}</h2>
            {generatorKind === "ladder" ? (
              <p style={{ color: "#475569" }}>
                Later rounds are created only after this round is saved or skipped.
              </p>
            ) : null}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
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
            const editable = canEditRound;
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
                  Court {match.court || "—"}
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
                      {match.scoreA == null || match.scoreB == null
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
              <button type="button" onClick={() => void saveRound()} disabled={busy} style={primaryButton}>
                {busy ? "Saving…" : "Save round scores"}
              </button>
              <input
                value={skipReason}
                onChange={(event_) => setSkipReason(event_.target.value)}
                placeholder="Optional skip reason"
                style={{ ...inputStyle, maxWidth: 260 }}
              />
              <button type="button" onClick={() => void skipRound()} disabled={busy} style={secondaryButton}>
                Skip round
              </button>
            </div>
          </div>
        ) : null}

        {round.status === "saved" ? (
          <section style={{ marginTop: "1rem", borderTop: "1px solid #e2e8f0", paddingTop: "1rem" }}>
            <h3>Round {roundNumber} results</h3>
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

        {round.status === "skipped" ? (
          <p style={{ marginTop: "1rem", padding: "0.7rem", background: "#fef3c7", borderRadius: "8px" }}>
            This round was skipped{round.skipReason ? `: ${round.skipReason}` : "."}
          </p>
        ) : null}

        {isCurrent && ["saved", "skipped"].includes(round.status) && session.status === "active" ? (
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
                : `Go to Round ${roundNumber + 1}`}
          </button>
        ) : null}
      </article>

      {isCurrent && session.status === "active" && Boolean(editToken) ? (
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

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Ratings and official results</h2>
        <p style={{ color: "#475569" }}>
          Public generator sessions are unrated. Share this page for viewing and score entry; staff can publish controlled official matches from the administrative generators.
        </p>
        <a href={`/api/clubs/${clubId}/play-generators/sessions/${sessionKey}/export?format=csv`}>
          Download session CSV
        </a>
      </article>

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

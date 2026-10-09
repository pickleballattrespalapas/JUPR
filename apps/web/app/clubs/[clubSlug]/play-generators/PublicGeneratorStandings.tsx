"use client";

import Link from "@/components/PublicClubLink";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import GeneratorSubmission, { GeneratorSubmissionStatus, generatorResultLabel } from "@/components/GeneratorSubmission";
import GeneratorPlayoff, { type PlayoffFormat, type GeneratorPlayoffEvent, type GeneratorPlayoffOptions } from "@/components/GeneratorPlayoff";
import PlayGeneratorStandingsTable, {
  PlayGeneratorStanding,
  standingsSortLabel
} from "@/components/PlayGeneratorStandingsTable";

type StandingsSort = "wins" | "points" | "differential";
type ScoringMode = "scored" | "unscored";

type Session = {
  session_key: string;
  title: string;
  status: string;
  version: string;
  generator_kind: string;
  play_format: string;
  rating_mode?: "rated" | "unrated";
  submission?: GeneratorSubmissionStatus | null;
  created_at?: string;
  scoring_mode?: ScoringMode;
  current_round_number?: number | null;
  total_rounds?: number | null;
  standings_sort?: StandingsSort;
  standings?: PlayGeneratorStanding[];
  playoff_options?: GeneratorPlayoffOptions | null;
  event: GeneratorPlayoffEvent & {
    scoringMode?: ScoringMode;
    standingsSort?: StandingsSort;
    currentRoundNumber?: number;
    totalRounds?: number;
    rounds?: Array<{ number: number; status: string; label?: string }>;
  };
};

type Props = { apiBase: string | null; clubId: string; sessionKey: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const linkButton = { display: "inline-flex", alignItems: "center", minHeight: "38px", padding: "0.45rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: "999px", color: "#0f172a", fontWeight: 800, textDecoration: "none" };
const primaryButton = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function operationKey(action: string): string { return `${action}-${Date.now()}-${Math.random().toString(16).slice(2)}`; }

function playFormatLabel(value: string): string {
  if (value === "singles") return "Singles";
  if (value === "doubles_singles") return "Doubles + Singles Mix";
  return "Doubles";
}

function sessionStatusLabel(status: string): string {
  return status === "completed" ? "Complete" : "In progress";
}

class UserFacingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UserFacingError";
  }
}

function requestFailureMessage(error: unknown, fallback: string): string {
  return error instanceof UserFacingError ? error.message : fallback;
}

function requestErrorMessage(status: number, detail?: unknown): string {
  if (status === 400 && typeof detail === "string" && /^(Save or skip|At least|This session already has a playoff|Submitted results|Playoff)/.test(detail)) return detail;
  if (detail === "Score entry is temporarily unavailable.") return "This play tool is temporarily unavailable.";
  if (detail === "Live sessions are temporarily unavailable. Please try again later.") return "This play tool is temporarily unavailable.";
  if (status === 403) return "This organizer link can’t make changes to this session.";
  if (status === 404) return "We couldn’t find this session.";
  if (status === 409) return "This session changed. Refresh the page and try again.";
  if (status === 429) return "Please wait a moment and try again.";
  return "We couldn’t complete that request. Please try again.";
}

export default function PublicGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {
  const router = useRouter();
  const [editToken, setEditToken] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("Loading standings…");
  const [busy, setBusy] = useState(false);

  async function loadSession(): Promise<void> {
    if (!apiBase) {
      setMessage("Standings are temporarily unavailable. Please try again.");
      return;
    }
    try {
      const response = await fetch(
        apiUrl(apiBase, `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),
        { cache: "no-store" }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new UserFacingError(requestErrorMessage(response.status, payload?.detail));
      if (payload?.session?.generator_kind !== "round_robin") throw new UserFacingError("This session doesn’t include round-robin standings.");
      setSession(payload.session as Session);
      setMessage("");
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t load the standings. Please try again."));
    }
  }

  useEffect(() => {
    const storageKey = `public-generator-edit:${clubId}:${sessionKey}`;
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";
    if (discovered) { sessionStorage.setItem(storageKey, discovered); setEditToken(discovered); }
    if (hash.has("edit")) window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);
    void loadSession();
  }, [apiBase, clubId, sessionKey]);

  async function updateSession(action: "advance" | "complete"): Promise<void> {
    if (!apiBase || !editToken || !session) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/${action}`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ edit_token: editToken, expected_version: Number(session.version), idempotency_key: operationKey(`standings-${action}`) })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new UserFacingError(requestErrorMessage(response.status, payload?.detail));
      const next = payload?.session as Session | undefined;
      if (!next) throw new UserFacingError("We couldn’t confirm the next round. Refresh the page and try again.");
      setSession(next);
      if (next.status === "completed") {
        setMessage("Session completed.");
        return;
      }
      const nextRound = Number(next.current_round_number || 1);
      router.push(`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);
      router.refresh();
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t continue the session. Please try again."));
    } finally {
      setBusy(false);
    }
  }

  async function startPlayoff(format: PlayoffFormat): Promise<void> {
    if (!apiBase || !editToken || !session || busy) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/playoff`),
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
          edit_token: editToken, expected_version: Number(session.version), playoff_format: format,
          idempotency_key: `generator-playoff:${sessionKey}:${session.version}:${format}`,
        }) }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new UserFacingError(requestErrorMessage(response.status, payload?.detail));
      if (!payload?.session) throw new UserFacingError("We couldn’t confirm the playoff. Refresh the page and try again.");
      setSession(payload.session as Session);
      router.push(`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${payload.session.current_round_number}`);
      router.refresh();
    } catch (error) {
      setMessage(requestFailureMessage(error, "We couldn’t start the playoff. Please try again."));
    } finally { setBusy(false); }
  }

  async function submitResults(organizerName: string, matchDate: string): Promise<void> {
    if (!apiBase || !editToken || !session) throw new Error("Use the organizer link to submit this session.");
    const response = await fetch(
      apiUrl(apiBase, `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/submit`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edit_token: editToken, idempotency_key: `generator-submit:${sessionKey}`, expected_version: Number(session.version), organizer_name: organizerName, match_date: matchDate })
      }
    );
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new UserFacingError(requestErrorMessage(response.status, payload?.detail));
    if (!payload?.session) throw new Error("Could not confirm submission. Please try again.");
    setSession(payload.session as Session);
    setMessage("");
  }

  if (!session) return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;

  const scoringMode = session.scoring_mode || session.event.scoringMode || "scored";
  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const totalRounds = Number(session.total_rounds || session.event.totalRounds || 1);
  const currentStatus = session.event.rounds?.find((row) => row.number === currentRound)?.status || "";
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);
  const resultsLocked = Boolean(session.submission);
  const canManage = Boolean(editToken) && scoringMode === "scored" && !resultsLocked;
  const canContinue = canManage && (session.status === "active" || (!session.event.playoff && session.status === "completed")) && ["saved", "skipped"].includes(currentStatus);
  const canFinish = canContinue && !session.event.playoff && session.status === "active" && visibleRounds.every((row) => ["saved", "skipped"].includes(row.status));
  const nextLabel = session.event.rounds?.find(row => row.number === currentRound + 1)?.label;
  const continueLabel = session.event.playoff
    ? currentRound >= totalRounds ? "Finish session" : `Continue to ${nextLabel || `Round ${currentRound + 1}`}`
    : session.status === "completed" ? `Keep playing · Round ${currentRound + 1}` : `Continue to Round ${currentRound + 1}`;

  if (scoringMode === "unscored") {
    return <article style={cardStyle}><h1>{session.title}</h1><p>This unscored Round-Robin does not use standings.</p><Link href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link></article>;
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr)", minWidth: 0, gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.4rem" }}><Link href={`/clubs/${clubId}/round-robin-generator`}>← Round-Robin Generator</Link></p>
        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>
        <p style={{ margin: 0, color: "#475569" }}>{playFormatLabel(session.play_format)} · {standingsSortLabel(sortMode)} · {sessionStatusLabel(session.status)} · {session.submission?.status !== "approved" ? `${session.rating_mode === "rated" ? "Rated" : "Unrated"} · ` : ""}{generatorResultLabel(session.submission)}</p>
      </article>
      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`} style={linkButton}>Current round</Link>
        {visibleRounds.map((row) => <Link key={row.number} href={`/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`} style={linkButton}>{row.label || `Round ${row.number}`}{row.status === "skipped" ? " · Skipped" : ""}</Link>)}
      </nav>
      {session.status === "completed" || session.submission ? (
        <div style={{ display: "grid", gap: "0.5rem" }}>
          <GeneratorSubmission ratingMode={session.rating_mode || "unrated"} submission={session.submission} canSubmit={!resultsLocked && session.status === "completed" && Boolean(editToken)} defaultDate={session.created_at} onSubmit={submitResults} />
          {session.submission ? <button type="button" onClick={() => void loadSession()} style={{ ...linkButton, cursor: "pointer", justifySelf: "start", background: "white" }}>Refresh approval status</button> : null}
        </div>
      ) : null}
      <GeneratorPlayoff event={session.event} options={session.playoff_options} canManage={Boolean(editToken)} locked={Boolean(session.submission)} busy={busy} onStart={startPlayoff} roundHref={round => `/clubs/${clubId}/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`} />
      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} title={session.event.playoff ? "Round-robin standings" : undefined} />
      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Final standings</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>
            The cumulative standings are preserved above.
          </p>
        </article>
      ) : null}
      {canContinue ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{session.status === "completed" ? "Keep playing" : continueLabel}</h2>
          <p style={{ color: "#475569" }}>{session.event.playoff ? "Save each playoff game, then continue to the final or finish the session." : "Play as many rounds as you like. Finish the session when you’re ready to submit results."}</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            <button type="button" onClick={() => void updateSession("advance")} disabled={busy} style={primaryButton}>{busy ? "Working…" : continueLabel}</button>
            {canFinish ? <button type="button" onClick={() => void updateSession("complete")} disabled={busy} style={{ ...linkButton, cursor: "pointer", background: "white" }}>Finish session</button> : null}
          </div>
        </article>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </div>
  );
}

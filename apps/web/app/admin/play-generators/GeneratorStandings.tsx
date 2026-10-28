"use client";

import GeneratorSessionDetails from "@/components/GeneratorSessionDetails";
import GeneratorSharing from "@/components/GeneratorSharing";
import { hubAutoAdvanceBoundary, type HubAutoAdvanceEvent } from "@/lib/hubAutoAdvanceBoundary";

import Link from "next/link";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { useRouter } from "next/navigation";
import GeneratorSubmission, { GeneratorSubmissionStatus, generatorResultLabel } from "@/components/GeneratorSubmission";
import GeneratorMoneyballExpectations, { type MoneyballExpectation, type MoneyballExpectationFormat } from "@/components/GeneratorMoneyballExpectations";
import GeneratorMoneyball, { type MoneyballSettlement } from "@/components/GeneratorMoneyball";
import GeneratorPlayoff, { type PlayoffFormat, type GeneratorPlayoffEvent, type GeneratorPlayoffOptions } from "@/components/GeneratorPlayoff";
import PlayGeneratorStandingsTable, {
  PlayGeneratorStanding,
  standingsSortLabel
} from "@/components/PlayGeneratorStandingsTable";
import { useAdminSession } from "@/lib/useAdminSession";
import { requestStandingsJson, StandingsRequestError, uncertainStandingsError, storedStandingsMutation, rememberStandingsMutation, type StandingsMutation } from "@/lib/generatorStandingsRequest";

type StandingsSort = "wins" | "points" | "differential" | "moneyball_expectations";
type ScoringMode = "scored" | "unscored";

type Session = {
  capabilities?: { can_manage: boolean; can_score: boolean; unavailable_reason: string | null };
  round_robin_hub_id?: string | null;
  exhibition_only?: boolean;
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
  moneyball?: MoneyballSettlement | null;
  standings_sort?: StandingsSort;
  standings?: PlayGeneratorStanding[];
  provisional_standings?: PlayGeneratorStanding[];
  standings_view?: {
    completed_through_round: number;
    scored_matches_in_partial_rounds: number;
    pending_matches: Array<{ match_id: string; round_number: number; side_a: string[]; side_b: string[] }>;
    pending_through_round: number;
    finish_after_round: number | null;
    final: boolean;
  };
  playoff_options?: GeneratorPlayoffOptions | null;
  official_publish?: { published_at?: string | null; published_match_ids?: number[] };
  event: GeneratorPlayoffEvent & HubAutoAdvanceEvent & {
    moneyballExpectations?: MoneyballExpectation[];
    moneyball?: MoneyballExpectationFormat;
    scoringMode?: ScoringMode;
    standingsSort?: StandingsSort;
    currentRoundNumber?: number;
    totalRounds?: number;
    rounds?: Array<{ number: number; status: string; label?: string }>;
  };
};

type Props = {
  apiBase: string | null; clubId: string; sessionKey: string;
  embedded?: boolean;
  onNavigate?: (target: { view: "round" | "standings"; roundNumber?: number }) => void;
  onSessionChanged?: () => void | Promise<void>;
  refreshKey?: string | number;
};

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

export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey, embedded = false, onNavigate, onSessionChanged, refreshKey }: Props) {
  const router = useRouter();
  const { accessToken } = useAdminSession();
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("Loading standings…");
  const [busy, setBusy] = useState(false);
  const [includePartial, setIncludePartial] = useState(false);
  const [pending, setPending] = useState<StandingsMutation | null>(null);
  const pendingRef = useRef<StandingsMutation | null>(null);
  const mutationRef = useRef(false);
  const context = JSON.stringify([apiBase, clubId, sessionKey]);
  const contextRef = useRef(context);
  const generationRef = useRef(0);
  const mountedRef = useRef(false);
  const readRef = useRef<AbortController | null>(null);
  const storageKey = `generator-standings-pending:${context}`;
  const blocked = busy || Boolean(pending);

  function currentRequest(generation: number): boolean {
    return mountedRef.current && contextRef.current === context && generationRef.current === generation;
  }
  function retainMutation(request: StandingsMutation | null): void {
    pendingRef.current = request;
    setPending(request);
    rememberStandingsMutation(storageKey, request);
  }

  function navigate(path: string): void {
    const round = path.match(/\/rounds\/(\d+)$/);
    if (onNavigate && round) { onNavigate({ view: "round", roundNumber: Number(round[1]) }); return; }
    router.push(path);
  }
  function refresh(): void { if (embedded) onSessionChanged?.(); else router.refresh(); }
  function captureSessionNavigation(click: MouseEvent<HTMLDivElement>): void {
    if (!onNavigate || click.button !== 0 || click.metaKey || click.ctrlKey || click.shiftKey || click.altKey) return;
    const anchor = (click.target as Element).closest?.("a");
    const href = anchor?.getAttribute("href") || "";
    if (!href.startsWith(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/`)) return;
    click.preventDefault(); click.stopPropagation(); navigate(href);
  }
  const loadRef = useRef<() => void>(() => {});
  loadRef.current = () => { if (!busy) void loadSession(); };
  const previousRefreshKey = useRef(refreshKey);
  useEffect(() => {
    if (previousRefreshKey.current === refreshKey) return;
    previousRefreshKey.current = refreshKey;
    if (refreshKey !== undefined) loadRef.current();
  }, [refreshKey]);

  async function loadSession(): Promise<void> {
    if (!apiBase || !accessToken || mutationRef.current) return;
    readRef.current?.abort();
    const controller = new AbortController();
    readRef.current = controller;
    const generation = ++generationRef.current;
    try {
      const payload = await requestStandingsJson<{ session: Session }>(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),
        { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store", signal: controller.signal }
      );
      if (!currentRequest(generation)) return;
      if (payload?.session?.generator_kind !== "round_robin") throw new Error("Standings are available for Round-Robin Generator sessions.");
      setSession(payload.session);
      setMessage("");
    } catch (error) {
      if (currentRequest(generation) && !controller.signal.aborted) setMessage(error instanceof Error ? error.message : "Unable to load standings.");
    }
  }

  useEffect(() => {
    mountedRef.current = true;
    contextRef.current = context;
    mutationRef.current = false;
    setBusy(false);
    setSession(null);
    setIncludePartial(false);
    setMessage("Loading standings…");
    const saved = storedStandingsMutation(storageKey);
    pendingRef.current = saved;
    setPending(saved);
    void loadSession();
    return () => {
      mountedRef.current = false;
      generationRef.current += 1;
      readRef.current?.abort();
    };
  }, [context, accessToken]);

  async function executeMutation(request: StandingsMutation): Promise<void> {
    if (!apiBase || !accessToken || mutationRef.current) return;
    if (pendingRef.current && (pendingRef.current.action !== request.action || pendingRef.current.body !== request.body)) {
      throw new Error("Retry the saved action before starting another change.");
    }
    mutationRef.current = true;
    readRef.current?.abort();
    const generation = ++generationRef.current;
    retainMutation(request);
    setBusy(true);
    setMessage("");
    try {
      const payload = await requestStandingsJson<{ session: Session }>(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/${request.action}`),
        { method: "POST", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" }, body: request.body }
      );
      if (!currentRequest(generation)) return;
      if (!payload?.session) throw new StandingsRequestError("The action was not confirmed. Retry the saved action to check its result.", 502);
      retainMutation(null);
      setSession(payload.session);
      if (request.action === "submit") { onSessionChanged?.(); return; }
      if (payload.session.status === "completed") { onSessionChanged?.(); setMessage("Session completed."); return; }
      navigate(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${payload.session.current_round_number || 1}`);
      refresh();
    } catch (error) {
      if (currentRequest(generation)) {
        if (!uncertainStandingsError(error)) retainMutation(null);
        setMessage(error instanceof Error ? error.message : "The result is not yet confirmed. Retry the saved action to check it.");
      }
      throw error;
    } finally {
      if (currentRequest(generation)) { mutationRef.current = false; setBusy(false); }
    }
  }

  async function updateSession(action: "advance" | "complete"): Promise<void> {
    if (!session || blocked) return;
    try { await executeMutation({ action, body: JSON.stringify({
      expected_version: session.version, idempotency_key: operationKey(`standings-${action}`),
      ...(action === "complete" ? { confirmation_text: "COMPLETE SESSION" } : {})
    }) }); } catch { /* The request's recovery controls retain the error. */ }
  }

  async function startPlayoff(format: PlayoffFormat): Promise<void> {
    if (!session || blocked) return;
    try { await executeMutation({ action: "playoff", body: JSON.stringify({
      expected_version: session.version, playoff_format: format,
      idempotency_key: `generator-playoff:${sessionKey}:${session.version.replace(/[^A-Za-z0-9._:-]/g, "_")}:${format}`
    }) }); } catch { /* The exact request remains retryable when uncertain. */ }
  }

  async function submitResults(organizerName: string, matchDate: string): Promise<void> {
    if (!apiBase || !accessToken || !session) throw new Error("Sign in to submit this session.");
    if (blocked) throw new Error("Resolve the pending action before submitting results.");
    await executeMutation({ action: "submit", body: JSON.stringify({
      expected_version: session.version, organizer_name: organizerName, match_date: matchDate
    }) });
  }

  if (!session) return <article style={cardStyle}><h1>Round-Robin standings</h1><p role="status">{message}</p><button type="button" onClick={() => void loadSession()} disabled={busy} style={linkButton}>Refresh standings</button>{pending ? <button type="button" style={linkButton} disabled={busy} onClick={() => void executeMutation(pending).catch(() => {})}>Retry saved action</button> : null}</article>;

  const scoringMode = session.scoring_mode || session.event.scoringMode || "scored";
  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const totalRounds = Number(session.total_rounds || session.event.totalRounds || 1);
  const currentStatus = session.event.rounds?.find((row) => row.number === currentRound)?.status || "";
  const moneyball = Boolean(session.moneyball);
  const moneyballDone = session.moneyball?.completedGames === session.moneyball?.totalGames;
  const view = session.standings_view;
  const showingPartial = includePartial && Boolean(view && session.provisional_standings);
  const shownStandings = showingPartial ? session.provisional_standings || [] : session.standings || [];
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);
  const resultsLocked = Boolean(session.submission || session.official_publish?.published_at || session.official_publish?.published_match_ids?.length);
  // Play-management capabilities do not govern the separate staff submission route.
  const canManage = Boolean(accessToken) && scoringMode === "scored" && !resultsLocked && session.capabilities?.can_manage !== false;
  const canContinue = canManage && !session.exhibition_only && !((moneyball || session.round_robin_hub_id) && session.status === "completed") && (session.status === "active" || (!session.event.playoff && session.status === "completed")) && ["saved", "skipped"].includes(currentStatus);
  const untouchedHubBoundary = hubAutoAdvanceBoundary(session.event) !== null;
  const canFinish = canManage && !session.exhibition_only && (!moneyball || moneyballDone) && !session.event.playoff && session.status === "active" && (untouchedHubBoundary || (canContinue && visibleRounds.every((row) => ["saved", "skipped"].includes(row.status))));
  const nextLabel = session.event.rounds?.find(row => row.number === currentRound + 1)?.label;
  const continueLabel = moneyball && moneyballDone ? "Finish Moneyball" : session.event.playoff
    ? currentRound >= totalRounds ? "Finish session" : `Continue to ${nextLabel || `Round ${currentRound + 1}`}`
    : session.status === "completed" ? `Keep playing · Round ${currentRound + 1}` : "Next scheduled round";

  if (scoringMode === "unscored") {
    return <div onClickCapture={captureSessionNavigation}><article style={cardStyle}><h1>{session.title}</h1><p>This unscored Round-Robin does not use standings.</p><Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link></article></div>;
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr)", minWidth: 0, gap: "1rem" }} onClickCapture={captureSessionNavigation}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        {!embedded ? <p style={{ margin: "0 0 0.4rem" }}><Link href="/admin/round-robin-generator">← Round-Robin Generator</Link></p> : null}
        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>
        <p style={{ margin: 0, color: "#475569" }}>{playFormatLabel(session.play_format)} · {standingsSortLabel(sortMode)} · {session.status === "completed" ? "Complete" : "In progress"} · {session.submission?.status !== "approved" ? `${session.rating_mode === "rated" ? "Rated" : "Unrated"} · ` : ""}{generatorResultLabel(session.submission)}</p>
              <GeneratorSessionDetails key={sessionKey} apiBase={apiBase} clubId={clubId} sessionKey={sessionKey} session={session} adminToken={accessToken || undefined} locked={Boolean(session.exhibition_only) || resultsLocked || !canManage} disabled={blocked} scored={scoringMode === "scored"} onSaved={next => { setSession(next); onSessionChanged?.(); }} />
      </article>
      {session.exhibition_only ? <aside style={{ ...cardStyle, background: "#fff7ed" }}><strong>Extra game · exhibition</strong><p>This game has its own result. It does not change either home round robin’s standings and cannot be submitted for ratings.</p></aside> : null}
      {session.round_robin_hub_id ? <aside style={{ ...cardStyle, background: "#eff6ff" }}>
        <strong>{view?.final ? "Final standings" : showingPartial ? "Provisional standings — all recorded games" : view ? `Completed rounds through Round ${view.completed_through_round}` : "Standings include completed rounds only."}</strong>
        {view && !view.final ? <>
          <p>{view.scored_matches_in_partial_rounds} recorded {view.scored_matches_in_partial_rounds === 1 ? "game is" : "games are"} in rounds that are still being completed.</p>
          <label><input type="checkbox" checked={includePartial} onChange={event => setIncludePartial(event.target.checked)} /> Include recorded games from unfinished rounds</label>
          <p>{view.finish_after_round !== null ? `Last round: ${view.finish_after_round}. ` : ""}{view.pending_matches.length} {view.pending_matches.length === 1 ? "game needs" : "games need"} a result through Round {view.pending_through_round}.</p>
          {view.pending_matches.length ? <details><summary>Show missing results</summary><ul>{view.pending_matches.map(match => <li key={`${match.round_number}:${match.match_id}`}>Round {match.round_number}: {match.side_a.join(" / ")} vs {match.side_b.join(" / ")}</li>)}</ul></details> : null}
        </> : !view ? <p>Scores saved for a partly finished round remain pending until every match in that round is resolved.</p> : null}
        {showingPartial && !view?.final ? <p>Provisional totals are for live tracking. Official results still use completed rounds.</p> : null}
        {!embedded ? <p><Link href={`/admin/round-robin-hub?hub=${encodeURIComponent(session.round_robin_hub_id)}`}>Open court board</Link></p> : null}
      </aside> : null}
      {session.capabilities?.unavailable_reason ? <p role="status">{session.capabilities.unavailable_reason}</p> : null}
      <GeneratorSharing apiBase={apiBase} clubId={clubId} sessionKey={sessionKey} version={session.version} adminToken={accessToken || undefined} locked={resultsLocked || !canManage} disabled={blocked} onChanged={async () => { await Promise.all([loadSession(), onSessionChanged?.()]); }} />
      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`} style={linkButton}>Current round</Link>
        {visibleRounds.map((row) => <Link key={row.number} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`} style={linkButton}>{row.label || `Round ${row.number}`}{row.status === "skipped" ? " · Skipped" : ""}</Link>)}
      </nav>
      {!session.exhibition_only && (session.status === "completed" || session.submission) ? (
        <div style={{ display: "grid", gap: "0.5rem" }}>
          <GeneratorSubmission ratingMode={session.rating_mode || "unrated"} submission={session.submission} canSubmit={!blocked && !resultsLocked && session.status === "completed" && Boolean(accessToken)} defaultDate={session.created_at} onSubmit={submitResults} />
          {session.submission ? <button type="button" onClick={() => void loadSession()} style={{ ...linkButton, cursor: "pointer", justifySelf: "start", background: "white" }}>Refresh approval status</button> : null}
          <p style={{ margin: 0 }}><Link href="/admin/play-generators/submissions">Review generator submissions</Link> (club administrators)</p>
        </div>
      ) : null}
      {!moneyball && !session.exhibition_only ? <GeneratorPlayoff event={session.event} options={session.playoff_options} canManage={canManage} locked={Boolean(session.submission) || Boolean(session.official_publish?.published_at) || Boolean(session.official_publish?.published_match_ids?.length)} busy={blocked} onStart={startPlayoff} roundHref={round => `/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`} /> : null}
      <GeneratorMoneyball value={session.moneyball} />
      {moneyball ? <GeneratorMoneyballExpectations rows={session.event.moneyballExpectations} locked format={session.event.moneyball} /> : null}
      <PlayGeneratorStandingsTable rows={shownStandings} sortMode={sortMode} title={session.event.playoff ? "Round-robin standings" : undefined} />
      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Session complete</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>
            The cumulative standings are preserved above.
          </p>
        </article>
      ) : null}
      {canContinue || canFinish ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{session.status === "completed" ? "Keep playing" : untouchedHubBoundary ? "Finish session" : continueLabel}</h2>
          <p style={{ color: "#475569" }}>{untouchedHubBoundary ? "The next round is queued. You can finish now before any of its matches are assigned." : session.event.playoff ? "Save each playoff game, then continue to the final or finish the session." : moneyball ? "Play the complete fixed rotation, then finish Moneyball to confirm the payout." : "Play as many rounds as you like. Finish the session when you’re ready to submit results."}</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            {canContinue ? <button type="button" onClick={() => void updateSession("advance")} disabled={blocked} style={primaryButton}>{busy ? "Working…" : continueLabel}</button> : null}
            {canFinish && !moneyball ? <button type="button" onClick={() => void updateSession("complete")} disabled={blocked} style={{ ...linkButton, cursor: "pointer", background: "white" }}>Finish session</button> : null}
          </div>
        </article>
      ) : null}
      {pending ? <aside style={cardStyle} aria-label="Unconfirmed standings action"><p>The saved {pending.action} action is not yet confirmed. Retry checks that exact action without creating a new one.</p><button type="button" style={linkButton} disabled={busy} onClick={() => void executeMutation(pending).catch(() => {})}>Retry saved action</button></aside> : null}
      <button type="button" onClick={() => void loadSession()} disabled={busy} style={linkButton}>Refresh standings</button>
      {message ? <p role="status">{message}</p> : null}
    </div>
  );
}

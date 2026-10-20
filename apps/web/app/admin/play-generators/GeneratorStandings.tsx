"use client";

import GeneratorSharing from "@/components/GeneratorSharing";

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

type StandingsSort = "wins" | "points" | "differential" | "moneyball_expectations";
type ScoringMode = "scored" | "unscored";

type Session = {
  round_robin_hub_id?: string | null;
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
  playoff_options?: GeneratorPlayoffOptions | null;
  official_publish?: { published_at?: string | null; published_match_ids?: number[] };
  event: GeneratorPlayoffEvent & {
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
  onSessionChanged?: () => void;
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
    if (!apiBase || !accessToken) return;
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}`),
        { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store" }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      if (payload?.session?.generator_kind !== "round_robin") throw new Error("Standings are available for Round-Robin Generator sessions.");
      setSession(payload.session as Session);
      setMessage("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load standings.");
    }
  }

  useEffect(() => { void loadSession(); }, [accessToken, apiBase, clubId, sessionKey]);

  async function updateSession(action: "advance" | "complete"): Promise<void> {
    if (!apiBase || !accessToken || !session) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/${action}`),
        {
          method: "POST",
          headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
          body: JSON.stringify({ expected_version: session.version, idempotency_key: operationKey(`standings-${action}`) })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      const next = payload?.session as Session | undefined;
      if (!next) throw new Error("Session advanced without a refreshed session.");
      setSession(next);
      onSessionChanged?.();
      if (next.status === "completed") {
        setMessage("Session completed.");
        return;
      }
      const nextRound = Number(next.current_round_number || 1);
      navigate(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);
      refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to continue the session.");
    } finally {
      setBusy(false);
    }
  }

  async function startPlayoff(format: PlayoffFormat): Promise<void> {
    if (!apiBase || !accessToken || !session || busy) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/playoff`),
        { method: "POST", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" }, body: JSON.stringify({
          expected_version: session.version, playoff_format: format,
          idempotency_key: `generator-playoff:${sessionKey}:${session.version.replace(/[^A-Za-z0-9._:-]/g, "_")}:${format}`,
        }) }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      if (!payload?.session) throw new Error("We couldn’t confirm the playoff. Refresh the page and try again.");
      setSession(payload.session as Session);
      onSessionChanged?.();
      navigate(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${payload.session.current_round_number}`);
      refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "We couldn’t start the playoff. Please try again.");
    } finally { setBusy(false); }
  }

  async function submitResults(organizerName: string, matchDate: string): Promise<void> {
    if (!apiBase || !accessToken || !session) throw new Error("Sign in to submit this session.");
    const response = await fetch(
      apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/submit`),
      {
        method: "POST",
        headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ expected_version: session.version, organizer_name: organizerName, match_date: matchDate })
      }
    );
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    if (!payload?.session) throw new Error("Could not confirm submission. Please try again.");
    setSession(payload.session as Session);
    onSessionChanged?.();
    setMessage("");
  }

  if (!session) return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;

  const scoringMode = session.scoring_mode || session.event.scoringMode || "scored";
  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const totalRounds = Number(session.total_rounds || session.event.totalRounds || 1);
  const currentStatus = session.event.rounds?.find((row) => row.number === currentRound)?.status || "";
  const moneyball = Boolean(session.moneyball);
  const moneyballDone = session.moneyball?.completedGames === session.moneyball?.totalGames;
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);
  const resultsLocked = Boolean(session.submission);
  const canManage = Boolean(accessToken) && scoringMode === "scored" && !resultsLocked;
  const canContinue = canManage && !(moneyball && session.status === "completed") && (session.status === "active" || (!session.event.playoff && session.status === "completed")) && ["saved", "skipped"].includes(currentStatus);
  const canFinish = canContinue && (!moneyball || moneyballDone) && !session.event.playoff && session.status === "active" && visibleRounds.every((row) => ["saved", "skipped"].includes(row.status));
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
      </article>
      {session.round_robin_hub_id ? <aside style={{ ...cardStyle, background: "#eff6ff" }}>
        <strong>Standings include completed rounds only.</strong>
        <p style={{ marginBottom: 0 }}>Scores saved for a partly finished round remain pending until every match in that round is resolved.
          {!embedded ? <> <Link href={`/admin/round-robin-hub?hub=${encodeURIComponent(session.round_robin_hub_id)}`}>Open court board</Link></> : null}
        </p>
      </aside> : null}
      <GeneratorSharing apiBase={apiBase} clubId={clubId} sessionKey={sessionKey} version={session.version} adminToken={accessToken || undefined} locked={resultsLocked} disabled={busy} onChanged={loadSession} />
      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`} style={linkButton}>Current round</Link>
        {visibleRounds.map((row) => <Link key={row.number} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`} style={linkButton}>{row.label || `Round ${row.number}`}{row.status === "skipped" ? " · Skipped" : ""}</Link>)}
      </nav>
      {session.status === "completed" || session.submission ? (
        <div style={{ display: "grid", gap: "0.5rem" }}>
          <GeneratorSubmission ratingMode={session.rating_mode || "unrated"} submission={session.submission} canSubmit={!resultsLocked && session.status === "completed" && Boolean(accessToken)} defaultDate={session.created_at} onSubmit={submitResults} />
          {session.submission ? <button type="button" onClick={() => void loadSession()} style={{ ...linkButton, cursor: "pointer", justifySelf: "start", background: "white" }}>Refresh approval status</button> : null}
          <p style={{ margin: 0 }}><Link href="/admin/play-generators/submissions">Review generator submissions</Link> (club administrators)</p>
        </div>
      ) : null}
      {!moneyball ? <GeneratorPlayoff event={session.event} options={session.playoff_options} canManage={Boolean(accessToken)} locked={Boolean(session.submission) || Boolean(session.official_publish?.published_at) || Boolean(session.official_publish?.published_match_ids?.length)} busy={busy} onStart={startPlayoff} roundHref={round => `/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`} /> : null}
      <GeneratorMoneyball value={session.moneyball} />
      {moneyball ? <GeneratorMoneyballExpectations rows={session.event.moneyballExpectations} locked format={session.event.moneyball} /> : null}
      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} title={session.event.playoff ? "Round-robin standings" : undefined} />
      {session.status === "completed" ? (
        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>
          <h2 style={{ marginTop: 0 }}>Session complete</h2>
          <p style={{ marginBottom: 0, color: "#166534" }}>
            The cumulative standings are preserved above.
          </p>
        </article>
      ) : null}
      {canContinue ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{session.status === "completed" ? "Keep playing" : continueLabel}</h2>
          <p style={{ color: "#475569" }}>{session.event.playoff ? "Save each playoff game, then continue to the final or finish the session." : moneyball ? "Play the complete fixed rotation, then finish Moneyball to confirm the payout." : "Play as many rounds as you like. Finish the session when you’re ready to submit results."}</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            <button type="button" onClick={() => void updateSession("advance")} disabled={busy} style={primaryButton}>{busy ? "Working…" : continueLabel}</button>
            {canFinish && !moneyball ? <button type="button" onClick={() => void updateSession("complete")} disabled={busy} style={{ ...linkButton, cursor: "pointer", background: "white" }}>Finish session</button> : null}
          </div>
        </article>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </div>
  );
}

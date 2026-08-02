"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PlayGeneratorStandingsTable, {
  PlayGeneratorStanding,
  standingsSortLabel
} from "@/components/PlayGeneratorStandingsTable";
import { useAdminSession } from "@/lib/useAdminSession";

type StandingsSort = "wins" | "points" | "differential";
type ScoringMode = "scored" | "unscored";

type Session = {
  session_key: string;
  title: string;
  status: string;
  version: string;
  generator_kind: string;
  play_format: string;
  scoring_mode?: ScoringMode;
  current_round_number?: number | null;
  total_rounds?: number | null;
  standings_sort?: StandingsSort;
  standings?: PlayGeneratorStanding[];
  event: {
    scoringMode?: ScoringMode;
    standingsSort?: StandingsSort;
    currentRoundNumber?: number;
    totalRounds?: number;
    rounds?: Array<{ number: number; status: string }>;
  };
};

type Props = { apiBase: string | null; clubId: string; sessionKey: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const linkButton = { display: "inline-flex", alignItems: "center", minHeight: "38px", padding: "0.45rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: "999px", color: "#0f172a", fontWeight: 800, textDecoration: "none" };
const primaryButton = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function operationKey(action: string): string { return `${action}-${Date.now()}-${Math.random().toString(16).slice(2)}`; }

export default function AdminGeneratorStandings({ apiBase, clubId, sessionKey }: Props) {
  const router = useRouter();
  const { accessToken } = useAdminSession();
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("Loading standings…");
  const [busy, setBusy] = useState(false);

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

  async function continueSession(): Promise<void> {
    if (!apiBase || !accessToken || !session) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch(
        apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/advance`),
        {
          method: "POST",
          headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
          body: JSON.stringify({ expected_version: session.version, idempotency_key: operationKey("standings-advance") })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      const next = payload?.session as Session | undefined;
      if (!next) throw new Error("Session advanced without a refreshed session.");
      setSession(next);
      if (next.status === "completed") {
        setMessage("Session completed.");
        return;
      }
      const nextRound = Number(next.current_round_number || 1);
      router.push(`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${nextRound}`);
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to continue the session.");
    } finally {
      setBusy(false);
    }
  }

  if (!session) return <article style={cardStyle}><h1>Round-Robin standings</h1><p>{message}</p></article>;

  const scoringMode = session.scoring_mode || session.event.scoringMode || "scored";
  const currentRound = Number(session.current_round_number || session.event.currentRoundNumber || 1);
  const totalRounds = Number(session.total_rounds || session.event.totalRounds || 1);
  const currentStatus = session.event.rounds?.find((row) => row.number === currentRound)?.status || "";
  const sortMode = session.standings_sort || session.event.standingsSort || "wins";
  const visibleRounds = (session.event.rounds || []).filter((row) => row.number <= currentRound);
  const canContinue = scoringMode === "scored" && session.status === "active" && ["saved", "skipped"].includes(currentStatus);

  if (scoringMode === "unscored") {
    return <article style={cardStyle}><h1>{session.title}</h1><p>This unscored Round-Robin does not use standings.</p><Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`}>Return to current round</Link></article>;
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.4rem" }}><Link href="/admin/round-robin-generator">← Round-Robin Generator</Link></p>
        <h1 style={{ margin: "0 0 0.35rem" }}>{session.title} standings</h1>
        <p style={{ margin: 0, color: "#475569" }}>{session.play_format === "singles" ? "Singles" : "Doubles"} · {standingsSortLabel(sortMode)} · {session.status}</p>
      </article>
      <nav aria-label="Round-Robin session navigation" style={{ ...cardStyle, display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <Link href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${currentRound}`} style={linkButton}>Current round</Link>
        {visibleRounds.map((row) => <Link key={row.number} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(sessionKey)}/rounds/${row.number}`} style={linkButton}>Round {row.number}</Link>)}
      </nav>
      <PlayGeneratorStandingsTable rows={session.standings || []} sortMode={sortMode} />
      {canContinue ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{currentRound >= totalRounds ? "Finish the session" : `Continue to Round ${currentRound + 1}`}</h2>
          <p style={{ color: "#475569" }}>The completed round results are included above. Continue when the organizer is ready for the next round.</p>
          <button type="button" onClick={() => void continueSession()} disabled={busy} style={primaryButton}>{busy ? "Continuing…" : currentRound >= totalRounds ? "Finish session" : `Continue to Round ${currentRound + 1}`}</button>
        </article>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </div>
  );
}

import Link from "next/link";
import { getClubPlayers } from "@/lib/api";
import { getAdminLeagueLiveStatus, getAdminLeagueManagerStatus, getAdminLeagueManagerApiBaseUrl } from "@/lib/adminLeagueManagerApi";
import { getAdminMatchUploaderStatus } from "@/lib/adminMatchUploaderApi";
import LeagueLiveRoundPanel from "./LeagueLiveRoundPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function LeagueManagerLivePage() {
  const clubSlug = "tres-palapas";
  const clubId = "tres_palapas";
  const [{ data: leagueStatus, error: leagueError }, { data: liveDomainStatus, error: liveDomainError }, { data: uploaderStatus, error: uploaderError }, { data: playersData, error: playersError }] = await Promise.all([
    getAdminLeagueManagerStatus(clubId),
    getAdminLeagueLiveStatus(clubId),
    getAdminMatchUploaderStatus(clubId),
    getClubPlayers(clubSlug)
  ]);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        League Manager Live
      </p>
      <h1 style={{ marginTop: 0 }}>League live round entry</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Streamlit-style league round workflow for one live round at a time: load league roster, arrange courts, generate Python-backed round-robin match slots, enter scores, and submit official league matches through the guarded Match Uploader path.
      </p>

      {leagueError ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {leagueError}</p> : null}
      {liveDomainError ? <p style={{ color: "#b91c1c" }}>Python League Live status is unavailable. {liveDomainError}</p> : null}
      {uploaderError ? <p style={{ color: "#b91c1c" }}>Match Uploader status is unavailable. {uploaderError}</p> : null}
      {playersError ? <p style={{ color: "#b91c1c" }}>Player lookup is unavailable. {playersError}</p> : null}

      <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Operational guardrails</h2>
        <ul style={{ color: "#475569", paddingLeft: "1.25rem" }}>
          <li>Schedule generation runs through FastAPI/Python Match Uploader preview logic.</li>
          <li>Official score submission runs through FastAPI/Python Match Uploader processing and audit logging.</li>
          <li>Only valid non-tied scores are submitted; later corrections should use Match Log and Replay History.</li>
          <li>Python owns deterministic roster seeding, bench selection, score aggregation, court movement, overrides, and next-round state; the browser never ranks players.</li>
          <li>Session snapshots and multi-round court movement use stale-version guards and idempotent operation keys so an interrupted night can be recovered safely.</li>
          <li>Keep Streamlit available for recovery until the persisted live workflow is proven in the staging pilot.</li>
        </ul>
      </article>

      {leagueStatus && liveDomainStatus && uploaderStatus ? (
        <LeagueLiveRoundPanel
          apiBase={getAdminLeagueManagerApiBaseUrl()}
          clubId={clubId}
          leagueStatus={leagueStatus}
          liveDomainStatus={liveDomainStatus}
          uploaderStatus={uploaderStatus}
          players={playersData?.players || []}
        />
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/league-manager">League Manager</Link> · <Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}

import Link from "next/link";
import { getClubPlayers } from "@/lib/api";
import { getAdminMatchUploaderApiBaseUrl, getAdminMatchUploaderStatus } from "@/lib/adminMatchUploaderApi";
import MatchUploaderForm from "./MatchUploaderForm";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminMatchUploaderPage() {
  const clubSlug = "tres-palapas";
  const clubId = "tres_palapas";
  const [{ data: playersData, error: playersError }, { data: status, error: statusError }] = await Promise.all([
    getClubPlayers(clubSlug),
    getAdminMatchUploaderStatus(clubId)
  ]);
  const players = playersData?.players ?? [];

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Match Uploader
      </p>
      <h1 style={{ marginTop: 0 }}>Match Uploader</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Enter singles, doubles batches, or one or more round robins. Search the club roster or create a player inline with a reviewed Starting JUPR.
      </p>

      {playersError ? <p style={{ color: "#b91c1c" }}>Player lookup is unavailable. {playersError}</p> : null}
      {statusError ? <p style={{ color: "#b91c1c" }}>Match Uploader status is unavailable. {statusError}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.status.replace(/_/g, " ")}</article>
          <article style={cardStyle}><strong>Max batch rows</strong><br />{status.max_batch_rows}</article>
          <article style={cardStyle}><strong>RR formats</strong><br />{status.round_robin_format_options?.length ?? 0}</article>
          <article style={cardStyle}><strong>Players loaded</strong><br />{players.length}</article>
        </div>
      ) : null}

      {!playersError && !players.length ? <p>No players are loaded yet. Create the first player directly in a match or round robin below.</p> : null}
      {status ? (
        <MatchUploaderForm
          apiBase={getAdminMatchUploaderApiBaseUrl()}
          clubId={clubId}
          players={players}
          status={status}
        />
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-log">Open Match Log</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}

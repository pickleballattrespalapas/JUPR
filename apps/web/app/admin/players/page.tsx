import { requireAdminWorkspace } from "@/lib/adminWorkspaceServer";
import Link from "next/link";
import { getAdminPlayerEditorApiBaseUrl, getAdminPlayerEditorStatus } from "@/lib/adminPlayerEditorApi";
import PlayerEditorPanel from "./PlayerEditorPanel";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function AdminPlayersPage() {
  const { clubId } = requireAdminWorkspace();
  const { data: status, error } = await getAdminPlayerEditorStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin Player Editor
      </p>
      <h1 style={{ marginTop: 0 }}>Player Editor</h1>
      <p style={{ color: "#334155", maxWidth: "880px" }}>
        Find a player to update their name, rating, or active status. You can also add players and manage individual league ratings.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Player Editor status is unavailable. {error}</p> : null}

      {status ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Status</strong><br />{status.enabled ? "Ready" : "Unavailable"}</article>
          <article style={cardStyle}><strong>Players</strong><br />{status.player_count ?? "—"}</article>
        </div>
      ) : null}

      {status ? <PlayerEditorPanel apiBase={getAdminPlayerEditorApiBaseUrl()} clubId={clubId} status={status} /> : null}

      <p style={{ marginTop: "1rem" }}>
        <Link href="/admin/match-uploader">Match Uploader</Link> · <Link href="/admin/match-log">Match Log</Link> · <Link href="/admin/replay-history">Replay History</Link> · <Link href="/admin">Operations cockpit</Link>
      </p>
    </section>
  );
}

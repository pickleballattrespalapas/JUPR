import Link from "next/link";
import { getClubPlayers, type PublicPlayer } from "@/lib/api";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function jupr(value?: number | null): string {
  if (value == null) return "—";
  return (Number(value) / 400).toFixed(3);
}

function record(player: PublicPlayer): string {
  return `${player.wins ?? 0}-${player.losses ?? 0}`;
}

export default async function TopPlayersPrintablePage({ searchParams }: { searchParams?: { limit?: string; active?: string } }) {
  const clubSlug = "tres-palapas";
  const limit = Math.max(5, Math.min(Number(searchParams?.limit || 50) || 50, 200));
  const activeOnly = String(searchParams?.active || "1") !== "0";
  const { data, error } = await getClubPlayers(clubSlug);
  const players = (data?.players || [])
    .filter((player) => !activeOnly || player.is_active !== false)
    .sort((left, right) => Number(right.rating || 0) - Number(left.rating || 0))
    .slice(0, limit);

  return (
    <section>
      <style>{`@media print { nav, header, footer, .no-print { display: none !important; } body { background: white !important; } article { break-inside: avoid; } table { font-size: 12px; } }`}</style>
      <p className="no-print" style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Top Active Players PDF
      </p>
      <h1 style={{ marginTop: 0 }}>Top active players</h1>
      <p style={{ color: "#334155" }}>
        Printable player ranking sheet for {data?.club?.name || "Tres Palapas"}. This page is browser-printable HTML and does not mutate data.
      </p>
      {error ? <p style={{ color: "#b91c1c" }}>Player list is unavailable. {error}</p> : null}

      <article className="no-print" style={{ ...cardStyle, marginBottom: "1rem" }}>
        <p style={{ marginTop: 0, color: "#475569" }}>Use the browser print dialog to save as PDF. Query options: <code>?limit=100</code> and <code>?active=0</code>.</p>
        <Link href="/admin">Operations cockpit</Link> · <Link href="/clubs/tres-palapas/leaderboards">Public leaderboard</Link>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Players shown: {players.length}</h2>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr><th align="left">Rank</th><th align="left">Player</th><th align="right">JUPR</th><th align="right">Record</th><th align="right">Matches</th><th align="left">Last played</th></tr>
          </thead>
          <tbody>
            {players.map((player, index) => (
              <tr key={player.id}>
                <td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{index + 1}</td>
                <td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td>
                <td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{jupr(player.rating)}</td>
                <td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{record(player)}</td>
                <td align="right" style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{player.matches_played ?? 0}</td>
                <td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{player.last_game_at ? String(player.last_game_at).slice(0, 10) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>
    </section>
  );
}

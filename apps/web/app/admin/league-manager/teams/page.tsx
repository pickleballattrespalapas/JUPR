import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import TeamLeaguesPanel from "./TeamLeaguesPanel";

export default async function AdminTeamLeaguesPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>Team leagues</h1>
      <LeagueManagerNav />
      <p style={{ color: "#334155", maxWidth: "900px" }}>Run fixed-partner weekly leagues with partner registration, optional substitutes, round robin scheduling, playoffs, standings, results, and recoverable corrections.</p>
      {error ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {error}</p> : null}
      {status ? <TeamLeaguesPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}
    </section>
  );
}

import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import LeagueRosterPanel from "./LeagueRosterPanel";

export default async function AdminLeagueRosterPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>League roster</h1>
      <LeagueManagerNav />
      <p style={{ color: "#334155", maxWidth: "880px" }}>Search the full club roster, select many players, and apply one recoverable league membership update.</p>
      {error ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {error}</p> : null}
      {status ? <LeagueRosterPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}
    </section>
  );
}

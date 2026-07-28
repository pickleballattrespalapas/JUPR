import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import LeagueSettingsPanel from "./LeagueSettingsPanel";

export default async function AdminLeagueSettingsPage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>League settings</h1>
      <LeagueManagerNav />
      <p style={{ color: "#334155", maxWidth: "880px" }}>Set the league schedule, courts, rules, ratings, and award defaults in one guided editor.</p>
      {error ? <p style={{ color: "#b91c1c" }}>League Manager status is unavailable. {error}</p> : null}
      {status ? <LeagueSettingsPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}
    </section>
  );
}

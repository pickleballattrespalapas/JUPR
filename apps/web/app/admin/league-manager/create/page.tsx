import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import LeagueCreatePanel from "./LeagueCreatePanel";

export default async function AdminLeagueCreatePage() {
  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>Start league setup</h1>
      <LeagueManagerNav managerOnly />
      <p style={{ color: "#334155", maxWidth: "720px" }}>
        Choose the league structure and create an inactive draft. You will then complete schedule, courts, match rules, awards, and playoffs in the league setup wizard before activation. League mode and match format cannot be casually converted later.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? <LeagueCreatePanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} /> : null}
    </section>
  );
}

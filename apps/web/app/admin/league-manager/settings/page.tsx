import { redirect } from "next/navigation";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import LeagueManagerNav from "../LeagueManagerNav";
import LeagueSettingsPanel from "./LeagueSettingsPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default async function AdminLeagueSettingsPage({ searchParams }: Props) {
  const leagueName = first(searchParams?.league).trim();
  const leagueType = first(searchParams?.mode).trim();
  if (!leagueName) redirect("/admin/league-manager");

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>{leagueName} settings</h1>
      <LeagueManagerNav leagueName={leagueName} leagueType={leagueType || null} />
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? <LeagueSettingsPanel apiBase={getAdminLeagueManagerApiBaseUrl()} clubId={clubId} status={status} initialLeague={leagueName} /> : null}
    </section>
  );
}

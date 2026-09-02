import { redirect } from "next/navigation";
import { getAdminLeagueManagerApiBaseUrl, getAdminLeagueManagerStatus } from "@/lib/adminLeagueManagerApi";
import { readLeagueRouteContext } from "@/lib/leagueRouteContext";
import LeagueHomePanel from "./LeagueHomePanel";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default async function AdminSelectedLeaguePage({ searchParams }: Props) {
  const context = readLeagueRouteContext(searchParams);
  if (!context.leagueId) redirect("/admin/league-manager");
  const leagueName = context.leagueName || context.leagueId;

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminLeagueManagerStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin League Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName}</h1>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>League Manager is unavailable. {error}</p> : null}
      {status ? (
        <LeagueHomePanel
          apiBase={getAdminLeagueManagerApiBaseUrl()}
          clubId={clubId}
          status={status}
          initialLeagueId={context.leagueId}
          initialLeague={leagueName}
          initialLeagueType={context.leagueType || null}
        />
      ) : null}
    </section>
  );
}

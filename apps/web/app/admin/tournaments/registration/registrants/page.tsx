import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentRegistrantListPanel from "./TournamentRegistrantListPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function TournamentRegistrantsPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Tournament Manager / Registration</p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} registrants</h1>
      <TournamentPhaseNav phase="registration" />
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Registration management is unavailable. {error}</p> : null}
      {data ? <TournamentRegistrantListPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} tournamentId={context.tournamentId} tournamentName={context.tournamentName || context.tournamentId} drawId={context.drawId} /> : null}
    </section>
  );
}

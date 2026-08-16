import { redirect } from "next/navigation";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentLifecycleOverviewPanel from "../TournamentLifecycleOverviewPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function TournamentRegistrationPhasePage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Registration
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} registration</h1>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Registration is unavailable. {error}</p> : null}
      {data ? (
        <TournamentLifecycleOverviewPanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={data}
          tournamentId={context.tournamentId}
          tournamentName={context.tournamentName || context.tournamentId}
          drawId={context.drawId}
          phase="registration"
        />
      ) : null}
    </section>
  );
}

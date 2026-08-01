import { redirect } from "next/navigation";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import TournamentLifecycleOverviewPanel from "../TournamentLifecycleOverviewPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default async function TournamentSetupPhasePage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Setup
      </p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} setup</h1>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Setup is unavailable. {error}</p> : null}
      {data ? (
        <TournamentLifecycleOverviewPanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={data}
          tournamentId={tournamentId}
          tournamentName={tournamentName || tournamentId}
          phase="setup"
        />
      ) : null}
    </section>
  );
}

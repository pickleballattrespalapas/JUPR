import { redirect } from "next/navigation";
import { Suspense } from "react";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import RegistrationManagementPanel from "./RegistrationManagementPanel";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

export const dynamic = "force-dynamic";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function AdminTournamentRegistrationManagementPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <Suspense fallback={null}>
        <TournamentPhaseNav phase="registration" />
      </Suspense>
      <p style={{ margin: "1rem 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament"} reports</h1>
      <p style={{ color: "#334155", maxWidth: "850px" }}>
        Filter registrations, download CSV reports, preview recipients, and review the guarded Operations import handoff.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament reports are unavailable. {error}</p> : null}
      {data ? (
        <Suspense fallback={<p aria-live="polite" style={{ color: "#64748b" }}>Loading registration reports...</p>}>
          <RegistrationManagementPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} initialTournamentId={context.tournamentId} initialTournamentName={context.tournamentName || context.tournamentId} initialDrawId={context.drawId} />
        </Suspense>
      ) : null}
    </section>
  );
}

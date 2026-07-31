import { redirect } from "next/navigation";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import SelectedTournamentPanelScope from "../SelectedTournamentPanelScope";
import BulkRegistrationPanel from "./BulkRegistrationPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default async function AdminTournamentBulkRegistrationPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} bulk actions</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Apply one reviewed registration, offline-payment, or admin-note update to multiple registrations.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament bulk actions are unavailable. {error}</p> : null}
      {data ? (
        <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
          <BulkRegistrationPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} />
        </SelectedTournamentPanelScope>
      ) : null}
    </section>
  );
}

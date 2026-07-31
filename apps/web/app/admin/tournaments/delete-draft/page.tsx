import { redirect } from "next/navigation";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import SelectedTournamentPanelScope from "../SelectedTournamentPanelScope";
import DeleteDraftPanel from "./DeleteDraftPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default async function AdminTournamentDeleteDraftPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#b91c1c", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>Delete {tournamentName || "tournament"} draft</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Delete only this unused draft shell. The API rechecks registrations, setup rows, draws, teams, games, and audit requirements before deletion.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Draft deletion is unavailable. {error}</p> : null}
      {data ? (
        <SelectedTournamentPanelScope tournamentId={tournamentId} tournamentName={tournamentName || null}>
          <DeleteDraftPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} />
        </SelectedTournamentPanelScope>
      ) : null}
    </section>
  );
}

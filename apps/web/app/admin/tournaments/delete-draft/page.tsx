import { redirect } from "next/navigation";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import DeleteDraftPanel from "./DeleteDraftPanel";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default async function AdminTournamentDeleteDraftPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#b91c1c", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>Delete {context.tournamentName || "tournament"} draft</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Delete only this unused draft shell. The API rechecks registrations, setup rows, draws, teams, games, and audit requirements before deletion.
      </p>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Draft deletion is unavailable. {error}</p> : null}
      {data ? (
        <DeleteDraftPanel apiBase={getAdminTournamentApiBaseUrl()} clubId={clubId} status={data} initialTournamentId={context.tournamentId} initialTournamentName={context.tournamentName || context.tournamentId} />
      ) : null}
    </section>
  );
}

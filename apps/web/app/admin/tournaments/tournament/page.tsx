import { redirect } from "next/navigation";
import { getAdminTournamentApiBaseUrl, getAdminTournamentStatus } from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentHomePanel from "./TournamentHomePanel";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default async function AdminSelectedTournamentPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager
      </p>
      <h1 style={{ marginTop: 0 }}>{context.tournamentName || "Tournament Home"}</h1>
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament Manager is unavailable. {error}</p> : null}
      {data ? (
        <TournamentHomePanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={data}
          tournamentId={context.tournamentId}
          initialName={context.tournamentName || null}
          initialDrawId={context.drawId || null}
        />
      ) : null}
    </section>
  );
}

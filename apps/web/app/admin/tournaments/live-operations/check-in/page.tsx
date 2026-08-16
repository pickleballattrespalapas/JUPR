import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import { getAdminTournamentApiBaseUrl } from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentCheckInPanel from "./TournamentCheckInPanel";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default function TournamentCheckInPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  return (
    <section style={{ minWidth: 0 }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Live Operations
      </p>
      <h1 style={{ marginTop: 0 }}>
        {context.tournamentName || "Tournament"} preflight and check-in
      </h1>
      <p style={{ color: "#475569", maxWidth: "72ch" }}>
        Confirm who is physically attending, verify the attending player&apos;s waiver,
        and resolve event-day blockers from durable tournament data.
      </p>
      <TournamentPhaseNav phase="live" />
      <TournamentCheckInPanel
        apiBase={getAdminTournamentApiBaseUrl()}
        clubId="tres_palapas"
        tournamentId={context.tournamentId}
      />
    </section>
  );
}

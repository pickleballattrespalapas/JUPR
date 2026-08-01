import { redirect } from "next/navigation";
import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import TeamTournamentAdminPanel from "./TeamTournamentAdminPanel";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default function TeamTournamentAdminPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Manager / Setup
      </p>
      <h1 style={{ marginTop: 0 }}>{tournamentName || "Tournament"} events and formats</h1>
      <TournamentPhaseNav phase="setup" />
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Choose one event format, reveal only its applicable rules, then review the summary before saving.
      </p>
      <TeamTournamentAdminPanel
        clubId="tres_palapas"
        initialTournamentId={tournamentId}
      />
    </section>
  );
}

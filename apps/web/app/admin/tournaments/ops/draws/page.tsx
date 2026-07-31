import { redirect } from "next/navigation";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }

export default function TournamentDrawOperationsPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="draws" kicker="Tournament Manager / Operations" title="draws, scoring, playoffs, and podiums" description="Create division draws, maintain teams, generate round-robin and playoff games, enter scores with optimistic concurrency, and generate or award podiums." tournamentId={tournamentId} tournamentName={tournamentName || null} />;
}

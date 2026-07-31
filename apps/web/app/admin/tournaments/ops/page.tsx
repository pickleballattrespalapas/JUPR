import { redirect } from "next/navigation";
import TournamentOpsWorkflowPage from "./TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default function AdminTournamentOpsPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="all" kicker="Tournament Manager / Operations" title="operations" description="Manage draws, team imports, reviewed results, scoring, playoffs, podiums, awards, and official match publication through recoverable audited operations." tournamentId={tournamentId} tournamentName={tournamentName || null} />;
}

import { redirect } from "next/navigation";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }

export default function TournamentTeamImportPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="import" kicker="Tournament Manager / Operations" title="registration and bulk team imports" description="Build draw teams from confirmed registration selections or reviewed CSV/TSV rows. Replace operations remain atomic and are blocked once games exist." tournamentId={tournamentId} tournamentName={tournamentName || null} />;
}

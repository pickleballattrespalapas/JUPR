import { redirect } from "next/navigation";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
export default function TournamentTeamImportPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="import" kicker="Tournament Manager / Operations" title="registration and bulk team imports" description="Build draw teams from confirmed registration selections or reviewed CSV/TSV rows. Replace operations remain atomic and are blocked once games exist." tournamentId={context.tournamentId} tournamentName={context.tournamentName || null} initialDrawId={context.drawId || null} />;
}

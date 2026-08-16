import { redirect } from "next/navigation";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentOpsWorkflowPage from "../../ops/TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentResultsImportPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="results" kicker="Tournament Manager / Publish" title="import results" description="Preview a DUPR-style CSV without writing, resolve every player and match decision, then commit only the exact reviewed fingerprint." tournamentId={context.tournamentId} tournamentName={context.tournamentName || null} initialDrawId={context.drawId || null} />;
}

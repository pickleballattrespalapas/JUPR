import { redirect } from "next/navigation";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };
export default async function TournamentDrawOperationsPage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  return (
    <TournamentOpsWorkflowPage
      workflow="draws"
      kicker="Tournament Manager / Live Operations"
      title="draw setup & recovery"
      description="Prepare division draws, maintain teams, generate or repair round-robin schedules, and cancel verified-empty draws or events before live play."
      tournamentId={context.tournamentId}
      tournamentName={context.tournamentName || null}
      initialDrawId={context.drawId || null}
    />
  );
}

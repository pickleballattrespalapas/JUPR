import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentLiveRoute from "../TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentCorrectionsRecoveryPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (context.dayId) {
    redirect(tournamentRouteHref("/admin/tournaments/live-operations", context, { panel: "corrections" }));
  }
  return <TournamentLiveRoute searchParams={searchParams} view="corrections" phase="live" kicker="Tournament Manager / Live Operations" title="corrections and recovery" description="Review scored games, confirm before-and-after corrections, and reconcile failed or uncertain draw operations." />;
}

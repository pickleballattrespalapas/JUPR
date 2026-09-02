import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentLiveRoute from "../live-operations/TournamentLiveRoute";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentOpsPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  const legacyImportTournamentId = String(Array.isArray(searchParams?.tournament_id) ? searchParams?.tournament_id[0] : searchParams?.tournament_id || "").trim();
  if (!searchParams?.tournament && legacyImportTournamentId) {
    redirect(tournamentRouteHref("/admin/tournaments/ops/import", context));
  }
  return <TournamentLiveRoute searchParams={searchParams} view="podium" phase="live" kicker="Tournament Manager / Live Operations" title="podium draft" description="Generate, explicitly review, and award the podium for the selected tournament draw." />;
}

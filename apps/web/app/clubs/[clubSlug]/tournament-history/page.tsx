import { notFound } from "next/navigation";
import PublicTournamentNav from "@/components/PublicTournamentNav";
import EventHistoryView from "@/components/EventHistoryView";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { EventHistory } from "@/lib/eventSeasons";

export default async function TournamentHistoryPage({ params, searchParams }: { params: { clubSlug: string }; searchParams: { tournament_id?: string; tournament?: string } }) {
  const event = searchParams.tournament_id || searchParams.tournament;
  if (!event) notFound();
  const history = await publicSiteFetch<EventHistory>(`/public/clubs/${encodeURIComponent(params.clubSlug)}/event-history?${new URLSearchParams({ kind: "tournament", event })}`);
  if (!history) notFound();
  const selected = history.seasons.find(season => season.selected);
  return <><h1>{history.series_name}</h1><PublicTournamentNav clubSlug={params.clubSlug} tournamentId={selected?.source_id} active="history" /><EventHistoryView history={history} /></>;
}

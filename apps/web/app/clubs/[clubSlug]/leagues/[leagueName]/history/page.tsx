import { notFound } from "next/navigation";
import PublicLeagueNav from "@/components/PublicLeagueNav";
import EventHistoryView from "@/components/EventHistoryView";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { EventHistory } from "@/lib/eventSeasons";

export default async function LeagueHistoryPage({ params }: { params: { clubSlug: string; leagueName: string } }) {
  const name = decodeURIComponent(params.leagueName);
  const history = await publicSiteFetch<EventHistory>(`/public/clubs/${encodeURIComponent(params.clubSlug)}/event-history?${new URLSearchParams({ kind: "league", event: name })}`);
  if (!history) notFound();
  const selected = history.seasons.find(season => season.selected);
  return <><h1>{history.series_name}</h1><PublicLeagueNav clubSlug={params.clubSlug} leagueName={name} active="history" team={selected?.league_type === "Team"} leagueView={selected?.complete ? "past" : "active"} /><EventHistoryView history={history} /></>;
}

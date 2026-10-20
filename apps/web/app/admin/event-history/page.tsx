import EventSeasonManager from "./EventSeasonManager";
import type { EventKind } from "@/lib/eventSeasons";
export default async function EventHistoryPage(props: { searchParams: Promise<{ kind?: string; event?: string }> }) {
  const searchParams = await props.searchParams;
  const kind = searchParams.kind;
  if (!["interclub", "league", "tournament"].includes(kind || "") || !searchParams.event) return <p>Open History from a league or tournament first.</p>;
  return <EventSeasonManager kind={kind as EventKind} sourceId={searchParams.event} />;
}

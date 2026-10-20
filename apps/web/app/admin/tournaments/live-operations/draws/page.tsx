import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function TournamentDrawsSchedulePage(props: Props) {
  const searchParams = await props.searchParams;
  redirect(tournamentRouteHref(
    "/admin/tournaments/live-operations",
    readTournamentRouteContext(searchParams),
    { panel: "draws" }
  ));
}

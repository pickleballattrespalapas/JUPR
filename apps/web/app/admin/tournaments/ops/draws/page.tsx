import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
export default function TournamentDrawOperationsPage({ searchParams }: Props) {
  redirect(tournamentRouteHref(
    "/admin/tournaments/live-operations",
    readTournamentRouteContext(searchParams),
    { panel: "draws" }
  ));
}

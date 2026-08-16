import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupRegistrationRulesPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  redirect(tournamentRouteHref("/admin/tournaments/setup/basics", context));
}

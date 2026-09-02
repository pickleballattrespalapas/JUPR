import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentLivePage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  redirect(tournamentRouteHref(
    "/admin/tournaments/live-operations",
    context,
    { panel: "queue", court: first(searchParams?.court), game: first(searchParams?.game) }
  ));
}

function first(value: string | string[] | undefined): string {
  return String(Array.isArray(value) ? value[0] || "" : value || "").trim();
}

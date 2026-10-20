import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function AdminTournamentLivePage(props: Props) {
  const searchParams = await props.searchParams;
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

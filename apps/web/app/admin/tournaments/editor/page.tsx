import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function AdminTournamentRegistrationEditorPage({ searchParams }: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  redirect(tournamentRouteHref("/admin/tournaments/registration/registrants", context));
}

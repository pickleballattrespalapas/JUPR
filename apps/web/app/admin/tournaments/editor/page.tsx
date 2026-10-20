import { redirect } from "next/navigation";
import { readTournamentRouteContext, tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function AdminTournamentRegistrationEditorPage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  redirect(tournamentRouteHref("/admin/tournaments/registration/registrants", context));
}

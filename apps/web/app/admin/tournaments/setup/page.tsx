import { redirect } from "next/navigation";
import { tournamentSetupStepHref } from "@/components/TournamentSetupWizardNav";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

// Keep the six-step wizard as the only selected-tournament Setup entry point.
export default async function TournamentSetupPhasePage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");
  redirect(
    tournamentSetupStepHref(
      "basics",
      context.tournamentId,
      context.tournamentName || context.tournamentId,
      context.drawId
    )
  );
}

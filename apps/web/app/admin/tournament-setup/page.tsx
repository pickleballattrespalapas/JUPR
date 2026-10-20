import { redirect } from "next/navigation";
import { tournamentSetupStepHref } from "@/components/TournamentSetupWizardNav";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

// Preserve the legacy URL but hand off immediately to the canonical builder.
export default async function TournamentSetupPage(props: Props) {
  const searchParams = await props.searchParams;
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments/create");
  redirect(
    tournamentSetupStepHref(
      "basics",
      context.tournamentId,
      context.tournamentName || context.tournamentId,
      context.drawId
    )
  );
}

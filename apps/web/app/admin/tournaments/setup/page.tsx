import { redirect } from "next/navigation";
import { tournamentSetupStepHref } from "@/components/TournamentSetupWizardNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

// Keep the six-step wizard as the only selected-tournament Setup entry point.
export default function TournamentSetupPhasePage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  redirect(
    tournamentSetupStepHref(
      "basics",
      tournamentId,
      tournamentName || tournamentId
    )
  );
}

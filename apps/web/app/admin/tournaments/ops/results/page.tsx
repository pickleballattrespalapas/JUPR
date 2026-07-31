import { redirect } from "next/navigation";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }

export default function TournamentResultsImportPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="results" kicker="Tournament Manager / Results" title="review and import results" description="Parse a DUPR-style CSV without writing, resolve each player and match decision, review the podium, and commit only the exact reviewed fingerprint." tournamentId={tournamentId} tournamentName={tournamentName || null} />;
}

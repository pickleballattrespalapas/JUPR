import { redirect } from "next/navigation";
import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };
function first(value: string | string[] | undefined): string { return Array.isArray(value) ? String(value[0] || "") : String(value || ""); }

export default function TournamentOfficialPublishPage({ searchParams }: Props) {
  const tournamentId = first(searchParams?.tournament).trim();
  const tournamentName = first(searchParams?.name).trim();
  if (!tournamentId) redirect("/admin/tournaments");
  return <TournamentOpsWorkflowPage workflow="publish" kicker="Tournament Manager / Official publish" title="publish official tournament matches" description="Publish finalized singles and doubles games through Match Log and rating processing with a separate staging gate, confirmation, email safety check, and replay-safe operation record." tournamentId={tournamentId} tournamentName={tournamentName || null} />;
}

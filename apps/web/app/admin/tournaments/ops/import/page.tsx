import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

export default function TournamentTeamImportPage() {
  return <TournamentOpsWorkflowPage workflow="import" kicker="Tournament Ops / Team imports" title="Registration and bulk team imports" description="Build draw teams from confirmed registration selections or reviewed CSV/TSV rows. Replace operations are atomic and remain blocked once games exist." />;
}

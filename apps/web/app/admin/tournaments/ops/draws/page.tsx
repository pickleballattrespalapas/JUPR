import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

export default function TournamentDrawOperationsPage() {
  return <TournamentOpsWorkflowPage workflow="draws" kicker="Tournament Ops / Draws" title="Draws, scoring, playoffs, and podiums" description="Create division draws, maintain teams, generate round-robin and playoff games, enter scores with optimistic concurrency, and generate or award podiums." />;
}

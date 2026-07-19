import TournamentOpsWorkflowPage from "./TournamentOpsWorkflowPage";

export default function AdminTournamentOpsPage() {
  return <TournamentOpsWorkflowPage workflow="all" kicker="Tournament Ops" title="Tournament operations cockpit" description="Guarded Next/FastAPI tournament operations for draws, team imports, reviewed results, scoring, playoffs, podiums, awards, and official match publication. Every staging write is permission-scoped, state-fingerprinted, recoverable, and audited." />;
}

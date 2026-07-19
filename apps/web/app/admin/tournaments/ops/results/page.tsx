import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

export default function TournamentResultsImportPage() {
  return <TournamentOpsWorkflowPage workflow="results" kicker="Tournament Ops / Results CSV" title="Review and import DUPR results" description="Parse a DUPR-style CSV without writing, resolve every player and match decision, review the podium, and commit only the exact reviewed fingerprint through the guarded FastAPI authority." />;
}

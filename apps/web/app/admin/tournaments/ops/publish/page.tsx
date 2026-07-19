import TournamentOpsWorkflowPage from "../TournamentOpsWorkflowPage";

export default function TournamentOfficialPublishPage() {
  return <TournamentOpsWorkflowPage workflow="publish" kicker="Tournament Ops / Official publish" title="Publish official tournament matches" description="Publish finalized singles and doubles games through Match Log and rating processing. This destructive boundary has its own staging gate, dual permission requirement, confirmation, email safety check, and replay-safe operation record." />;
}

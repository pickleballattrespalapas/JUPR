import TournamentPhaseNav from "@/components/TournamentPhaseNav";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentStatus
} from "@/lib/adminTournamentApi";
import TournamentOpsPanel, { type OpsWorkflow } from "./TournamentOpsPanel";

type Props = {
  workflow: OpsWorkflow;
  kicker: string;
  title: string;
  description: string;
  tournamentId: string;
  tournamentName?: string | null;
  initialDrawId?: string | null;
};

export default async function TournamentOpsWorkflowPage({
  workflow,
  kicker,
  title,
  description,
  tournamentId,
  tournamentName,
  initialDrawId = null
}: Props) {
  const clubId = "tres_palapas";
  const { data, error } = await getAdminTournamentStatus(clubId);
  const phase = workflow === "results" || workflow === "publish" ? "publish" : "live";

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {kicker}
      </p>
      <h1 style={{ marginTop: 0 }}>
        {tournamentName ? `${tournamentName} — ${title}` : title}
      </h1>
      <TournamentPhaseNav phase={phase} />
      <p style={{ color: "#334155", maxWidth: "860px" }}>{description}</p>
      {error ? (
        <p role="alert" style={{ color: "#b91c1c" }}>
          Tournament Operations are unavailable. {error}
        </p>
      ) : null}
      {data ? (
        <TournamentOpsPanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={data}
          workflow={workflow}
          initialTournamentId={tournamentId}
          initialDrawId={initialDrawId}
        />
      ) : null}
    </section>
  );
}

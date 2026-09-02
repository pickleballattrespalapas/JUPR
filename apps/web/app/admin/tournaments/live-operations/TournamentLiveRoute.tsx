import { redirect } from "next/navigation";
import TournamentPhaseNav, { type TournamentPhase } from "@/components/TournamentPhaseNav";
import TournamentLivePanel, { type TournamentOperatorView } from "@/app/admin/tournament-live/TournamentLivePanel";
import {
  getAdminTournamentApiBaseUrl,
  getAdminTournamentLiveStatus
} from "@/lib/adminTournamentApi";
import { readTournamentRouteContext } from "@/lib/tournamentRouteContext";

type SearchParams = Record<string, string | string[] | undefined>;

type Props = {
  searchParams?: SearchParams;
  view: TournamentOperatorView;
  phase: TournamentPhase;
  kicker: string;
  title: string;
  description: string;
};

export default async function TournamentLiveRoute({
  searchParams,
  view,
  phase,
  kicker,
  title,
  description
}: Props) {
  const context = readTournamentRouteContext(searchParams);
  if (!context.tournamentId) redirect("/admin/tournaments");

  const clubId = "tres_palapas";
  const { data: status, error } = await getAdminTournamentLiveStatus(clubId);
  const heading = context.tournamentName ? `${context.tournamentName} ${title}` : title;

  return (
    <section style={{ minWidth: 0, maxWidth: "100%" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>{kicker}</p>
      <h1 style={{ marginTop: 0 }}>{heading}</h1>
      <p style={{ color: "#334155", maxWidth: "850px" }}>{description}</p>
      <TournamentPhaseNav phase={phase} />
      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament operations are unavailable. {error}</p> : null}
      {status ? (
        <TournamentLivePanel
          apiBase={getAdminTournamentApiBaseUrl()}
          clubId={clubId}
          status={status}
          initialTournamentId={context.tournamentId}
          initialTournamentName={context.tournamentName || null}
          initialDrawId={context.drawId}
          initialDayId={context.dayId}
          view={view}
        />
      ) : null}
    </section>
  );
}

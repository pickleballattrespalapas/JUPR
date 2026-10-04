import CompetitionWorkspace from "./CompetitionWorkspace";

export default function CompetitionPage({ searchParams }: { searchParams: { season?: string; meet?: string } }) {
  return <CompetitionWorkspace initialSeasonId={typeof searchParams.season === "string" ? searchParams.season : ""} initialMeetId={typeof searchParams.meet === "string" ? searchParams.meet : ""} />;
}

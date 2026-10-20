import CompetitionWorkspace from "./CompetitionWorkspace";

export default async function CompetitionPage(props: { searchParams: Promise<{ season?: string; meet?: string }> }) {
  const searchParams = await props.searchParams;
  return <CompetitionWorkspace initialSeasonId={typeof searchParams.season === "string" ? searchParams.season : ""} initialMeetId={typeof searchParams.meet === "string" ? searchParams.meet : ""} />;
}

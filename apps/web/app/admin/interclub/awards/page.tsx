import SeasonAwards from "./SeasonAwards";
export default async function SeasonAwardsPage(props: { searchParams: Promise<{ season?: string }> }) {
  const searchParams = await props.searchParams;
  return <SeasonAwards seasonId={typeof searchParams.season === "string" ? searchParams.season : ""} />;
}

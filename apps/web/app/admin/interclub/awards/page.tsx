import SeasonAwards from "./SeasonAwards";
export default function SeasonAwardsPage({ searchParams }: { searchParams: { season?: string } }) {
  return <SeasonAwards seasonId={typeof searchParams.season === "string" ? searchParams.season : ""} />;
}

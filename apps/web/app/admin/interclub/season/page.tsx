import SeasonGuide from "./SeasonGuide";

export default function SeasonGuidePage({ searchParams }: { searchParams: { season?: string } }) {
  return <SeasonGuide seasonId={typeof searchParams.season === "string" ? searchParams.season : ""} />;
}

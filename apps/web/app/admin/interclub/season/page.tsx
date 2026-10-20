import SeasonGuide from "./SeasonGuide";

export default async function SeasonGuidePage(props: { searchParams: Promise<{ season?: string }> }) {
  const searchParams = await props.searchParams;
  return <SeasonGuide seasonId={typeof searchParams.season === "string" ? searchParams.season : ""} />;
}

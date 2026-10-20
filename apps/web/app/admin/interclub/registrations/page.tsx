import RegistrationWorkspace from "./RegistrationWorkspace";

export default async function RegistrationsPage(
  props: { searchParams: Promise<{ season?: string; meet?: string; step?: string }> }
) {
  const searchParams = await props.searchParams;
  return <RegistrationWorkspace initialSeasonId={typeof searchParams.season === "string" ? searchParams.season : ""}
    initialMeetId={typeof searchParams.meet === "string" ? searchParams.meet : ""}
    initialStep={searchParams.step === "availability" || searchParams.step === "lineups" ? searchParams.step : "pool"} />;
}

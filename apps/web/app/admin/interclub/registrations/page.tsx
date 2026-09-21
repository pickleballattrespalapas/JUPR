import RegistrationWorkspace from "./RegistrationWorkspace";

export default function RegistrationsPage({ searchParams }: { searchParams: { season?: string; meet?: string; step?: string } }) {
  return <RegistrationWorkspace initialSeasonId={typeof searchParams.season === "string" ? searchParams.season : ""}
    initialMeetId={typeof searchParams.meet === "string" ? searchParams.meet : ""}
    initialStep={searchParams.step === "availability" || searchParams.step === "lineups" ? searchParams.step : "pool"} />;
}

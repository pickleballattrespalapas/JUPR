import RegistrationWorkspace from "./RegistrationWorkspace";

export default function RegistrationsPage({ searchParams }: { searchParams: { season?: string } }) {
  return <RegistrationWorkspace initialSeasonId={typeof searchParams.season === "string" ? searchParams.season : ""} />;
}

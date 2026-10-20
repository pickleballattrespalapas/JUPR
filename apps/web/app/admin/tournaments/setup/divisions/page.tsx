import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

// Divisions are a distinct guided step after event-family setup.
export default async function TournamentSetupDivisionsPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentSetupWizardPage step="divisions" searchParams={searchParams} />;
}

import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

// Divisions are a distinct guided step after event-family setup.
export default function TournamentSetupDivisionsPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="divisions" searchParams={searchParams} />;
}

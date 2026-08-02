import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = {
  searchParams?: Record<string, string | string[] | undefined>;
};

export default function TournamentSetupDivisionsPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="divisions" searchParams={searchParams} />;
}

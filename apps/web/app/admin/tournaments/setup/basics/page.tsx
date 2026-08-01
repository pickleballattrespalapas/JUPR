import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupBasicsPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="basics" searchParams={searchParams} />;
}

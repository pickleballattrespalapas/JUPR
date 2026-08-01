import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupSchedulePage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="schedule" searchParams={searchParams} />;
}

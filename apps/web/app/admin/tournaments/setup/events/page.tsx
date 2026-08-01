import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupEventsPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="events" searchParams={searchParams} />;
}

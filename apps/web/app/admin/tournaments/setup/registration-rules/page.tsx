import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupRegistrationRulesPage({ searchParams }: Props) {
  return (
    <TournamentSetupWizardPage
      step="registration-rules"
      searchParams={searchParams}
    />
  );
}

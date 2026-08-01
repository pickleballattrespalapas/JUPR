import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupReviewPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="review" searchParams={searchParams} />;
}

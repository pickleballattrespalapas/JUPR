import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

export default function TournamentSetupPricingPage({ searchParams }: Props) {
  return <TournamentSetupWizardPage step="pricing" searchParams={searchParams} />;
}

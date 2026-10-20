import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function TournamentSetupPricingPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentSetupWizardPage step="pricing" searchParams={searchParams} />;
}

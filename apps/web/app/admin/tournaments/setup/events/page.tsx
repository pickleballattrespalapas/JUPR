import TournamentSetupWizardPage from "../TournamentSetupWizardPage";

type Props = { searchParams?: Promise<Record<string, string | string[] | undefined>> };

export default async function TournamentSetupEventsPage(props: Props) {
  const searchParams = await props.searchParams;
  return <TournamentSetupWizardPage step="events" searchParams={searchParams} />;
}

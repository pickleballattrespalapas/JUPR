import Link from "@/components/PublicClubLink";
import { teamLeagueApiBaseUrl } from "@/lib/teamLeagueApi";
import PartnerConfirmationPanel from "./PartnerConfirmationPanel";

type Props = {
  params: Promise<{ clubSlug: string }>;
  searchParams: Promise<{ team?: string }>;
};

export default async function TeamLeaguePartnerConfirmationPage(props: Props) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Team league</p>
      <h1>Confirm your partner</h1>
      <PartnerConfirmationPanel apiBase={teamLeagueApiBaseUrl()} clubSlug={params.clubSlug} teamId={String(searchParams.team || "")} />
      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${params.clubSlug}/team-leagues`}>View team leagues</Link></p>
    </section>
  );
}

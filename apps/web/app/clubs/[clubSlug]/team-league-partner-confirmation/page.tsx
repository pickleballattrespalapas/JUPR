import Link from "next/link";
import { teamLeagueApiBaseUrl } from "@/lib/teamLeagueApi";
import PartnerConfirmationPanel from "./PartnerConfirmationPanel";

type Props = {
  params: { clubSlug: string };
  searchParams: { team?: string };
};

export default function TeamLeaguePartnerConfirmationPage({ params, searchParams }: Props) {
  return (
    <section>
      <p style={{ color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Team league</p>
      <h1>Confirm your partner</h1>
      <PartnerConfirmationPanel apiBase={teamLeagueApiBaseUrl()} clubSlug={params.clubSlug} teamId={String(searchParams.team || "")} />
      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${params.clubSlug}/team-leagues`}>View team leagues</Link></p>
    </section>
  );
}

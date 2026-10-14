import SharedGeneratorLanding from "@/components/SharedGeneratorLanding";

type Props = { params: { clubSlug: string; sessionKey: string } };
export default function SharedRoundRobinPage({ params }: Props) {
  return <SharedGeneratorLanding apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null} clubSlug={params.clubSlug} sessionKey={params.sessionKey} />;
}

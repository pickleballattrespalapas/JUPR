import SharedGeneratorLanding from "@/components/SharedGeneratorLanding";

type Props = { params: Promise<{ clubSlug: string; sessionKey: string }> };
export default async function SharedRoundRobinPage(props: Props) {
  const params = await props.params;
  return <SharedGeneratorLanding apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null} clubSlug={params.clubSlug} sessionKey={params.sessionKey} />;
}

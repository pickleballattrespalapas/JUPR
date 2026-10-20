import PartnerInvitationResponse from "./PartnerInvitationResponse";

export const metadata = { title: "Partner request | PCS", robots: { index: false, follow: false }, referrer: "no-referrer" };

export default async function PartnerRequestPage(props: { params: Promise<{ clubSlug: string }> }) {
  const params = await props.params;
  return <PartnerInvitationResponse clubSlug={params.clubSlug} />;
}

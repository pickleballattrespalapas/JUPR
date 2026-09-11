import PartnerInvitationResponse from "./PartnerInvitationResponse";

export const metadata = { title: "Partner request | PCS", robots: { index: false, follow: false }, referrer: "no-referrer" };

export default function PartnerRequestPage({ params }: { params: { clubSlug: string } }) {
  return <PartnerInvitationResponse clubSlug={params.clubSlug} />;
}

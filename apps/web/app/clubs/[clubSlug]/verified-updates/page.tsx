import VerifiedUpdatesPageContent from "@/app/verified-updates/VerifiedUpdatesPageContent";

export default function ClubVerifiedUpdatesPage({ params, searchParams }: { params: { clubSlug: string }; searchParams?: { player_id?: string; pid?: string } }) {
  const initialPlayerId = searchParams?.player_id || searchParams?.pid || null;
  return <VerifiedUpdatesPageContent clubSlug={params.clubSlug} initialPlayerId={initialPlayerId} />;
}

import VerifiedUpdatesPageContent from "@/app/verified-updates/VerifiedUpdatesPageContent";

export default async function ClubVerifiedUpdatesPage(
  props: { params: Promise<{ clubSlug: string }>; searchParams?: Promise<{ player_id?: string; pid?: string }> }
) {
  const searchParams = await props.searchParams;
  const params = await props.params;
  const initialPlayerId = searchParams?.player_id || searchParams?.pid || null;
  return <VerifiedUpdatesPageContent clubSlug={params.clubSlug} initialPlayerId={initialPlayerId} />;
}

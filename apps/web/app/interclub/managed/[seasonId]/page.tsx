import { notFound } from "next/navigation";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { PublicLeague } from "@/lib/interclubPublic";
import PublicInterclubLeague from "@/components/PublicInterclubLeague";
export const metadata = { robots: { index: false, follow: false } };
export default async function ManagedLeaguePage({ params }: { params: Promise<{ seasonId: string }> }) {
  const { seasonId } = await params;
  if (!/^[0-9a-f-]{36}$/i.test(seasonId)) notFound();
  const league = await publicSiteFetch<PublicLeague>(`/public/managed-interclub/${seasonId}`);
  if (!league) notFound();
  return <PublicInterclubLeague league={league} showHistory={false} />;
}

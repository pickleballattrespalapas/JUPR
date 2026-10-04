import { notFound } from "next/navigation";
import Link from "next/link";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { PublicLeague } from "@/lib/interclubPublic";
import PublicInterclubLeague from "@/components/PublicInterclubLeague";
import type { EventHistory } from "@/lib/eventSeasons";
export const metadata = {
  robots: { index: false, follow: false },
};
export default async function LeaguePage({
  params,
  searchParams,
}: {
  params: { seasonId: string };
  searchParams: { view?: string };
}) {
  if (!/^[0-9a-f-]{36}$/i.test(params.seasonId)) notFound();
  const [league, history] = await Promise.all([
    publicSiteFetch<PublicLeague>(`/public/interclub/${params.seasonId}`),
    publicSiteFetch<EventHistory>(`/public/interclub/${params.seasonId}/history`).catch(() => null),
  ]);
  if (!league) notFound();
  return (
    <>
      <p>
        <Link href="/clubs">Find a club</Link>
      </p>
      <PublicInterclubLeague league={league} history={history} initialView={searchParams.view === "history" ? "history" : searchParams.view === "results" ? "results" : "standings"} />
    </>
  );
}

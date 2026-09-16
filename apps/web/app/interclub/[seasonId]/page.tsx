import { notFound } from "next/navigation";
import Link from "next/link";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { PublicLeague } from "@/lib/interclubPublic";
import PublicInterclubLeague from "@/components/PublicInterclubLeague";
export const metadata = {
  title: "Interclub league · PCS",
  robots: { index: false, follow: false },
};
export default async function LeaguePage({
  params,
}: {
  params: { seasonId: string };
}) {
  if (!/^[0-9a-f-]{36}$/i.test(params.seasonId)) notFound();
  const league = await publicSiteFetch<PublicLeague>(
    `/public/interclub/${params.seasonId}`,
  );
  if (!league) notFound();
  return (
    <>
      <p>
        <Link href="/clubs">Find a club</Link>
      </p>
      <PublicInterclubLeague league={league} />
    </>
  );
}

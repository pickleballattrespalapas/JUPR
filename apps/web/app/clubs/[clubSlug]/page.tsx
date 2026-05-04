import Link from "next/link";
import { getClub } from "@/lib/api";

type ClubPageProps = {
  params: { clubSlug: string };
};

export default async function ClubPage({ params }: ClubPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClub(clubSlug);

  return (
    <section>
      <h1>Club: {data?.name ?? clubSlug}</h1>
      {error ? (
        <p style={{ color: "#b91c1c" }}>Could not load club data. {error}</p>
      ) : (
        <>
          <p style={{ marginBottom: "0.25rem" }}><strong>Slug:</strong> {data?.slug}</p>
          {data?.tagline ? <p style={{ margin: "0.25rem 0" }}>{data.tagline}</p> : null}
        </>
      )}
      <p>
        <Link href={`/clubs/${clubSlug}/leaderboards`}>View public leaderboards</Link>
      </p>
    </section>
  );
}

import Link from "next/link";
import { getClub } from "@/lib/api";

type ClubPageProps = {
  params: { clubSlug: string };
};

export default async function ClubPage({ params }: ClubPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClub(clubSlug);

  if (error) {
    return (
      <section>
        <h1>Club: {clubSlug}</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this club right now. {error}</p>
        <p>
          <Link href={`/clubs/${clubSlug}/leaderboards`}>Try the public leaderboards</Link>
        </p>
      </section>
    );
  }

  return (
    <section>
      <h1>{data?.name ?? clubSlug}</h1>
      <p style={{ marginBottom: "0.25rem" }}><strong>Slug:</strong> {data?.slug ?? clubSlug}</p>
      {data?.tagline ? <p style={{ margin: "0.25rem 0" }}>{data.tagline}</p> : null}
      {data?.support_email ? (
        <p style={{ margin: "0.25rem 0" }}>
          <strong>Support:</strong> <a href={`mailto:${data.support_email}`}>{data.support_email}</a>
        </p>
      ) : null}
      <p>
        <Link href={`/clubs/${clubSlug}/leaderboards`}>View public leaderboards</Link>
      </p>
    </section>
  );
}

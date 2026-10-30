import PublishedClubShell from "@/components/PublishedClubShell";
import { Suspense } from "react";
import { requestShareMetadata } from "@/lib/shareMetadata";
export async function generateMetadata(props: { params: Promise<{ clubSlug: string }> }) {
  const params = await props.params;
  return requestShareMetadata(`/clubs/${params.clubSlug}`);
}
export default function ClubLayout(
  props: {
    params: Promise<{ clubSlug: string }>;
    children: React.ReactNode;
  }
) {
  return <Suspense fallback={<p role="status">Opening club page…</p>}><PublishedClubShell {...props} /></Suspense>;
}

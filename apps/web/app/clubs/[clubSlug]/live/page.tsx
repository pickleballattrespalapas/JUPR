import { redirect } from "next/navigation";

type Props = { params: Promise<{ clubSlug: string }> };

export default async function RetiredPublicLiveLanding(props: Props) {
  const params = await props.params;
  redirect(`/clubs/${params.clubSlug}/play`);
}

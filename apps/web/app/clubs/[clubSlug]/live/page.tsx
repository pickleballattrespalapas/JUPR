import { redirect } from "next/navigation";

type Props = { params: { clubSlug: string } };

export default function RetiredPublicLiveLanding({ params }: Props) {
  redirect(`/clubs/${params.clubSlug}/play`);
}

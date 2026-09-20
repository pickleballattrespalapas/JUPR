import type { ReactNode } from "react";
import { requestShareMetadata } from "@/lib/shareMetadata";

export async function generateMetadata({ params }: { params: { clubSlug: string } }) {
  return requestShareMetadata(`/clubs/${params.clubSlug}`);
}
export default function ClubLayout({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

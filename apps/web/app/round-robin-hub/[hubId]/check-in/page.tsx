import type { Metadata } from "next";
import PublicHubCheckin from "./PublicHubCheckin";

export const metadata: Metadata = {
  title: "Event check-in | Pickleball Club Sandwich",
  robots: { index: false, follow: false },
};

export default async function HubCheckinPage({ params }: { params: Promise<{ hubId: string }> }) {
  const { hubId } = await params;
  return <PublicHubCheckin hubId={hubId}
    apiBase={process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null} />;
}

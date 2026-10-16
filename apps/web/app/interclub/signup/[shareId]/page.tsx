import type { Metadata } from "next";
import SeasonSignup from "./SeasonSignup";

export const metadata: Metadata = {
  title: "Join your club’s interclub player pool | PCS",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

export default function SeasonSignupPage({ params }: { params: { shareId: string } }) {
  return <SeasonSignup key={params.shareId} shareId={params.shareId} />;
}

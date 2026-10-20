import type { Metadata } from "next";
import SeasonSignup from "./SeasonSignup";

export const metadata: Metadata = {
  title: "Join your club’s interclub player pool | PCS",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

export default async function SeasonSignupPage(props: { params: Promise<{ shareId: string }> }) {
  const params = await props.params;
  return <SeasonSignup key={params.shareId} shareId={params.shareId} />;
}

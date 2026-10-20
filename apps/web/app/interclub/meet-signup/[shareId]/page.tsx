import type { Metadata } from "next";
import MeetSignup from "../MeetSignup";
export const metadata: Metadata = { title: "Sign up to play an interclub meet | PCS", robots: { index: false, follow: false }, referrer: "no-referrer" };
export default async function Page(props: { params: Promise<{ shareId: string }> }) {
  const params = await props.params;
  return <MeetSignup key={params.shareId} shareId={params.shareId} />;
}

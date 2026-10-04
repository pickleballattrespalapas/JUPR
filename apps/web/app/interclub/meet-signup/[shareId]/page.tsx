import type { Metadata } from "next";
import MeetSignup from "../MeetSignup";
export const metadata: Metadata = { title: "Sign up to play an interclub meet | PCS", robots: { index: false, follow: false }, referrer: "no-referrer" };
export default function Page({ params }: { params: { shareId: string } }) { return <MeetSignup key={params.shareId} shareId={params.shareId} />; }

import type { Metadata } from "next";
import MeetSignup from "../MeetSignup";
export const metadata: Metadata = { title: "Your interclub meet signup | PCS", robots: { index: false, follow: false }, referrer: "no-referrer" };
export default function Page() { return <MeetSignup />; }

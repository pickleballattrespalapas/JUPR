import { Suspense } from "react";
import AcceptInvitation from "./AcceptInvitation";

export default function AcceptInvitationPage() {
  return <Suspense fallback={<p>Loading invitation…</p>}><AcceptInvitation /></Suspense>;
}

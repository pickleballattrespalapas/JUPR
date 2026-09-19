import type { Metadata } from "next";
import PlayerResponse from "./PlayerResponse";

export const metadata: Metadata = {
  title: "Your interclub availability | PCS",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

export default function PlayerResponsePage() {
  return <PlayerResponse />;
}

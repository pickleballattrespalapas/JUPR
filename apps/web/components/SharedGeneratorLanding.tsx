"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function SharedGeneratorLanding({ apiBase, clubSlug, sessionKey }: { apiBase: string | null; clubSlug: string; sessionKey: string }) {
  const router = useRouter();
  const [message, setMessage] = useState("Opening the current round…");
  useEffect(() => {
    let cancelled = false;
    async function open() {
      try {
        if (!apiBase) throw new Error();
        const response = await fetch(`${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/shared-play-generators/sessions/${encodeURIComponent(sessionKey)}`, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok || !payload.session) throw new Error();
        if (!cancelled) router.replace(`/clubs/${encodeURIComponent(clubSlug)}/round-robin-generator/shared/${encodeURIComponent(sessionKey)}/rounds/${payload.session.current_round_number || 1}${window.location.hash}`);
      } catch {
        if (!cancelled) setMessage("We couldn’t open this round robin. Check the link with the organizer.");
      }
    }
    void open();
    return () => { cancelled = true; };
  }, [apiBase, clubSlug, sessionKey, router]);
  return <p role="status">{message}</p>;
}

"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { publicTeamLeagueHref } from "@/lib/teamLeagueLinks";

export default function TeamLeagueRegistrationLink({ clubSlug, leagueName }: {
  clubSlug: string;
  leagueName: string;
}) {
  const path = publicTeamLeagueHref(clubSlug, leagueName);
  const [origin, setOrigin] = useState("");
  const [feedback, setFeedback] = useState<{ path: string; message: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { setOrigin(window.location.origin); }, []);

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(new URL(path, window.location.origin).href);
      setFeedback({ path, message: "Registration link copied." });
    } catch {
      input.current?.focus();
      input.current?.select();
      setFeedback({ path, message: "Select and copy the registration link above." });
    }
  }

  return (
    <article style={{ border: "1px solid #bfdbfe", borderRadius: "14px", padding: "1rem", background: "#eff6ff", minWidth: 0 }}>
      <h2 style={{ marginTop: 0 }}>Registration link</h2>
      <p>Share this league page with players. Signup is available there when online registration is open.</p>
      <label style={{ display: "block" }}>
        <strong>Player signup link</strong>
        <input ref={input} readOnly value={`${origin}${path}`} onFocus={(event) => event.target.select()}
          style={{ display: "block", boxSizing: "border-box", width: "100%", minWidth: 0, padding: "0.65rem", border: "1px solid #94a3b8", borderRadius: "8px", font: "inherit", marginTop: "0.35rem" }} />
      </label>
      <div style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap", alignItems: "center", marginTop: "0.75rem" }}>
        <button type="button" onClick={() => void copyLink()} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", font: "inherit", fontWeight: 800, cursor: "pointer" }}>Copy registration link</button>
        <Link href={path} target="_blank" rel="noopener noreferrer">Open registration page</Link>
      </div>
      {feedback?.path === path ? <p role="status" style={{ marginBottom: 0 }}>{feedback.message}</p> : null}
    </article>
  );
}

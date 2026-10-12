"use client";

import { useEffect, useRef } from "react";

type Snapshot = {
  clubSlug: string;
  tournamentId: string;
  active: boolean;
  flow: "new" | "edit";
  stage: string;
  name: string;
  email: string;
  error: string | null;
  completed?: boolean;
  selections: { event_option_id: string; partner_mode?: string; partner_name?: string; same_email_as_player?: boolean }[];
};

export function useRegistrationAttempt(snapshot: Snapshot): string {
  const attempt = useRef("");
  const latest = useRef(snapshot);
  const lastSent = useRef("");
  const activeScope = useRef("");
  latest.current = snapshot;
  // Lazily initialize in the browser, including when a submit happens before an effect.
  const scope = `${snapshot.clubSlug}\u0000${snapshot.tournamentId}\u0000${snapshot.flow}`;
  if (typeof window !== "undefined" && snapshot.active && activeScope.current !== scope) {
    attempt.current = crypto.randomUUID(); activeScope.current = scope; lastSent.current = "";
  }
  if (!snapshot.active) activeScope.current = "";

  function send(event: "PROGRESS" | "CLIENT_REJECTED" | "LEFT_FORM") {
    const current = latest.current;
    if (!current.active || current.completed || typeof window === "undefined") return;
    const base = process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
    if (!base || !attempt.current) return;
    const body = JSON.stringify({
      attempt_id: attempt.current, tournament_id: current.tournamentId,
      event, flow: current.flow, stage: current.stage,
      display_name: current.name.slice(0, 160), email: current.email.slice(0, 320),
      message: event === "CLIENT_REJECTED" ? (current.error || "").slice(0, 500) : "",
      selections: current.selections.slice(0, 8)
    });
    if (event !== "LEFT_FORM" && lastSent.current === body) return;
    lastSent.current = body;
    // A failed support request cannot interrupt the registration itself.
    void fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(current.clubSlug)}/tournament-registration/attempts`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body, keepalive: true, credentials: "omit"
    }).catch(() => undefined);
  }

  const signature = JSON.stringify([snapshot.active, snapshot.flow, snapshot.stage, snapshot.name, snapshot.email, snapshot.error, snapshot.selections, snapshot.completed]);
  useEffect(() => {
    if (typeof window === "undefined" || typeof document === "undefined") return;
    const timer = setTimeout(() => send(latest.current.error ? "CLIENT_REJECTED" : "PROGRESS"), snapshot.error ? 0 : 750);
    return () => clearTimeout(timer);
    // The serialized safe snapshot is the complete dependency; send reads its latest ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);
  useEffect(() => {
    if (typeof window === "undefined" || typeof document === "undefined") return;
    const onHide = () => { if (document.visibilityState === "hidden") send("LEFT_FORM"); };
    const onLeave = () => send("LEFT_FORM");
    window.addEventListener("pagehide", onLeave);
    document.addEventListener("visibilitychange", onHide);
    return () => {
      onLeave();
      window.removeEventListener("pagehide", onLeave);
      document.removeEventListener("visibilitychange", onHide);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot.clubSlug, snapshot.tournamentId]);
  return attempt.current;
}

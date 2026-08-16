"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentLifecycle,
  AdminTournamentLiveSnapshotResponse,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard
} from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  tournamentId: string;
  initialName?: string | null;
  initialDrawId?: string | null;
};

type PhaseCard = {
  title: string;
  state: string;
  description: string;
  href: string;
  note: string;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};
const phaseCardStyle = {
  ...cardStyle,
  display: "grid",
  gap: "0.45rem",
  alignContent: "start",
  textDecoration: "none",
  color: "#0f172a"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function dateValue(value?: string | null): string {
  return value ? String(value).slice(0, 10) : "";
}

function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string,
  drawId = ""
): string {
  return tournamentRouteHref(path, { tournamentId, tournamentName, drawId });
}

function phaseStateStyle(state: string) {
  if (["Complete", "Ready", "Open"].includes(state)) {
    return { color: "#166534", background: "#dcfce7", borderColor: "#bbf7d0" };
  }
  if (["Blocked", "Needs attention"].includes(state)) {
    return { color: "#991b1b", background: "#fee2e2", borderColor: "#fecaca" };
  }
  if (["In progress", "Live"].includes(state)) {
    return { color: "#92400e", background: "#fef3c7", borderColor: "#fde68a" };
  }
  return { color: "#475569", background: "#f8fafc", borderColor: "#cbd5e1" };
}

function phaseCards(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  lifecycle: AdminTournamentLifecycle | null,
  initialDrawId: string
): PhaseCard[] {
  const registrationStatus = String(
    detail.tournament.registration_status || ""
  ).toLowerCase();
  const setupReady = Boolean(
    detail.tournament.start_date &&
      detail.tournament.end_date &&
      detail.days.length &&
      detail.event_options.length
  );
  const registrations = detail.summary.registrations || 0;
  const registrationOpen = registrationStatus === "open";
  const counts = lifecycle?.counts;
  const selectedDrawId = initialDrawId || lifecycle?.draws[0]?.draw_id || "";
  const gamesExist = Boolean(counts?.games);
  const liveInProgress = Boolean(gamesExist && ((counts?.open_games || 0) > 0 || (counts?.finalized_games || 0) > 0));
  const publishReady = lifecycle?.domain_readiness.official_publish.ready;
  const publishBlockers = lifecycle?.domain_readiness.official_publish.blockers || [];

  return [
    {
      title: "Setup",
      state: setupReady ? "Ready" : "In progress",
      description:
        "Basics, events and formats, registration rules, pricing, extras, schedule, courts, and final review.",
      href: selectedHref(
        "/admin/tournaments/setup",
        tournamentId,
        tournamentName,
        selectedDrawId
      ),
      note: setupReady
        ? `${detail.event_options.length} events across ${detail.days.length} tournament days`
        : "Complete dates, events, and tournament days"
    },
    {
      title: "Registration",
      state: registrationOpen
        ? "Open"
        : registrations
          ? "In progress"
          : "Not started",
      description:
        "Registrants, partners and teams, offline payments, extras, communications, and reports.",
      href: selectedHref(
        "/admin/tournaments/registration",
        tournamentId,
        tournamentName,
        selectedDrawId
      ),
      note: `${registrations} registration${registrations === 1 ? "" : "s"}`
    },
    {
      title: "Live Operations",
      description:
        "Preflight, check-in, draws, court schedule, live scoring, corrections, recovery, and podium draft.",
      href: selectedHref(
        "/admin/tournaments/live-operations",
        tournamentId,
        tournamentName,
        selectedDrawId
      ),
      state: !lifecycle ? "Needs attention" : liveInProgress ? "In progress" : setupReady && registrations ? "Ready" : "Blocked",
      note: !lifecycle
        ? "Authoritative draw state is unavailable; readiness is not inferred"
        : gamesExist
        ? `${counts?.finalized_games || 0} of ${counts?.games || 0} games scored; ${counts?.open_games || 0} open`
        : setupReady && registrations
          ? "Core setup and registrations are present"
          : "Complete setup and add registrations first"
    },
    {
      title: "Publish",
      state: publishReady ? ((counts?.published_games || 0) === (counts?.games || 0) && Boolean(counts?.games) ? "Complete" : "Ready") : "Blocked",
      description:
        "Review results, publish ready divisions, create official matches, complete replay, and close the tournament.",
      href: selectedHref(
        "/admin/tournaments/publish",
        tournamentId,
        tournamentName,
        selectedDrawId
      ),
      note: !lifecycle
        ? "Authoritative publish readiness is unavailable; publishing remains blocked"
        : publishReady
        ? "Results become official only through Publish"
        : publishBlockers.map((blocker) => blocker.message).join(" · ") || "Authoritative publish readiness is unavailable"
    }
  ];
}

function nextAction(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  lifecycle: AdminTournamentLifecycle | null,
  initialDrawId: string
): { label: string; href: string; reason: string } {
  const counts = lifecycle?.counts;
  const drawId = initialDrawId || lifecycle?.draws[0]?.draw_id || "";
  if (!lifecycle) {
    return {
      label: "Review authoritative Live state",
      href: selectedHref("/admin/tournaments/live-operations", tournamentId, tournamentName, drawId),
      reason: "Authoritative draw state is unavailable. Live and publish readiness will not be inferred from setup or registration counts."
    };
  }
  if ((counts?.games || 0) > 0 && (counts?.open_games || 0) > 0) {
    return {
      label: "Continue scoring",
      href: selectedHref("/admin/tournament-live", tournamentId, tournamentName, drawId),
      reason: `${counts?.finalized_games || 0} of ${counts?.games || 0} games scored; ${counts?.open_games || 0} open.`
    };
  }
  if ((counts?.games || 0) > 0 && !lifecycle?.domain_readiness.official_publish.ready) {
    return {
      label: "Resolve publish blockers",
      href: selectedHref("/admin/tournaments/ops/results", tournamentId, tournamentName, drawId),
      reason: lifecycle?.domain_readiness.official_publish.blockers.map((blocker) => blocker.message).join(" ") || "Publishing prerequisites remain incomplete."
    };
  }
  const unpublishedGames = counts?.unpublished_games ?? Math.max(0, (counts?.games || 0) - (counts?.published_games || 0));
  if (lifecycle?.domain_readiness.official_publish.ready && unpublishedGames > 0) {
    return {
      label: "Publish ready divisions",
      href: selectedHref("/admin/tournaments/ops/publish", tournamentId, tournamentName, drawId),
      reason: "Every tournament prerequisite is complete; official publication is ready for deliberate review."
    };
  }
  if (!detail.tournament.start_date || !detail.tournament.end_date) {
    return {
      label: "Finish tournament basics",
      href: selectedHref(
        "/admin/tournaments/setup/basics",
        tournamentId,
        tournamentName,
        drawId
      ),
      reason: "Tournament dates are incomplete."
    };
  }
  if (!detail.event_options.length || !detail.days.length) {
    return {
      label: "Continue setup",
      href: selectedHref(
        "/admin/tournaments/setup/events",
        tournamentId,
        tournamentName,
        drawId
      ),
      reason: "Events or tournament days still need attention."
    };
  }
  if (String(detail.tournament.registration_status || "").toLowerCase() !== "open") {
    return {
      label: "Review and open registration",
      href: selectedHref(
        "/admin/tournaments/setup/review",
        tournamentId,
        tournamentName,
        drawId
      ),
      reason: "Core setup is ready; registration is not open."
    };
  }
  if (!detail.summary.registrations) {
    return {
      label: "Manage registration",
      href: selectedHref(
        "/admin/tournaments/registration",
        tournamentId,
        tournamentName,
        drawId
      ),
      reason: "Registration is open and no entrants are recorded yet."
    };
  }
  return {
    label: "Open Live Operations",
    href: selectedHref(
      "/admin/tournaments/live-operations",
      tournamentId,
      tournamentName,
      drawId
    ),
    reason: "Authoritative lifecycle state is available; review the live preflight before proceeding."
  };
}

export default function TournamentHomePanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  initialName,
  initialDrawId
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [lifecycle, setLifecycle] = useState<AdminTournamentLifecycle | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    clearProtectedState
  );

  function clearProtectedState() {
    setDetail(null);
    setLifecycle(null);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before opening this tournament.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(String(payload?.detail || `API error (${response.status})`));
    }
    return payload as T;
  }

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const [payload, liveSnapshot] = await Promise.all([
        requestJson<AdminTournamentDetailResponse>(
          `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`
        ),
        requestJson<AdminTournamentLiveSnapshotResponse>(
          `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/snapshot${initialDrawId ? `?draw_id=${encodeURIComponent(initialDrawId)}` : ""}`
        ).catch(() => null)
      ]);
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setLifecycle(liveSnapshot?.lifecycle || null);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to load tournament home."
        );
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status.enabled ? `${accessToken}\u0000${tournamentId}` : "",
    loadDetail
  );

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        Tournament Manager is currently unavailable.
      </article>
    );
  }
  if (sessionLoading && !accessToken) {
    return <p role="status">Loading tournament workspace…</p>;
  }
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  const tournamentName = detail?.tournament.name || initialName || tournamentId;
  const routeDrawId = String(initialDrawId || "");
  const next = detail ? nextAction(detail, tournamentId, tournamentName, lifecycle, routeDrawId) : null;
  const phases = detail
    ? phaseCards(detail, tournamentId, tournamentName, lifecycle, routeDrawId)
    : [];

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {message ? (
        <p
          role="status"
          style={{ color: /unable|error|required|reload|cannot/i.test(message) ? "#b91c1c" : "#166534" }}
        >
          {message}
        </p>
      ) : null}
      {busy && !detail ? <p role="status">Loading {tournamentName}…</p> : null}

      {detail ? (
        <>
          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
              <div>
                <h2 style={{ margin: 0 }}>{tournamentName}</h2>
                <p style={{ color: "#475569", marginBottom: 0 }}>
                  {dateValue(detail.tournament.start_date) || "Date not set"} – {dateValue(detail.tournament.end_date) || "Date not set"}
                </p>
              </div>
              <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                <span><strong>{detail.summary.registrations}</strong> registrations</span>
                <span><strong>{detail.event_options.length}</strong> events</span>
                <span><strong>{detail.days.length}</strong> days</span>
              </div>
            </div>
          </article>

          {next ? (
            <article style={{ ...cardStyle, borderColor: "#93c5fd" }}>
              <p style={{ marginTop: 0, color: "#475569" }}><strong>Next action</strong></p>
              <h2 style={{ marginTop: 0 }}>{next.label}</h2>
              <p style={{ color: "#475569" }}>{next.reason}</p>
              <Link href={next.href} style={{ display: "inline-block", padding: "0.6rem 0.9rem", borderRadius: "999px", background: "#0f172a", color: "white", textDecoration: "none", fontWeight: 800 }}>
                {next.label}
              </Link>
            </article>
          ) : null}

          {lifecycle && !lifecycle.domain_readiness.official_publish.ready ? (
            <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
              <h2 style={{ marginTop: 0 }}>Publish blockers</h2>
              <ul>
                {lifecycle.domain_readiness.official_publish.blockers.map((blocker) => <li key={`${blocker.code}:${blocker.draw_id || "tournament"}`}>{blocker.message}</li>)}
              </ul>
            </article>
          ) : null}

          {!lifecycle ? (
            <article style={{ ...cardStyle, borderColor: "#fde68a", background: "#fffbeb" }}>
              <h2 style={{ marginTop: 0 }}>Authoritative tournament state unavailable</h2>
              <p>Live and publish readiness remain blocked until the draw lifecycle can be loaded. Setup and registration counts are not used as a substitute.</p>
              <Link href={selectedHref("/admin/tournaments/live-operations", tournamentId, tournamentName, routeDrawId)}>Open Live Operations and reload</Link>
            </article>
          ) : null}

          <section aria-label="Tournament lifecycle phases">
            <h2>Tournament workflow</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.85rem" }}>
              {phases.map((phase) => (
                <Link key={phase.title} href={phase.href} style={phaseCardStyle}>
                  <span style={{ width: "fit-content", border: "1px solid", borderRadius: "999px", padding: "0.15rem 0.5rem", fontSize: "0.78rem", fontWeight: 800, ...phaseStateStyle(phase.state) }}>
                    {phase.state}
                  </span>
                  <strong>{phase.title}</strong>
                  <span style={{ color: "#475569" }}>{phase.description}</span>
                  <small style={{ color: "#64748b" }}>{phase.note}</small>
                </Link>
              ))}
            </div>
          </section>

          <article style={cardStyle}>
  <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
    <div>
      <h2 style={{ marginTop: 0 }}>Tournament basics</h2>
      <p style={{ color: "#475569", marginBottom: 0 }}>
        {tournamentName} · {dateValue(detail.tournament.start_date) || "Date not set"} – {dateValue(detail.tournament.end_date) || "Date not set"}
      </p>
    </div>
    <Link
      href={selectedHref("/admin/tournaments/setup/basics", tournamentId, tournamentName, routeDrawId)}
      style={{ display: "inline-block", padding: "0.6rem 0.9rem", borderRadius: "999px", background: "#0f172a", color: "white", textDecoration: "none", fontWeight: 800 }}
    >
      Edit in guided setup
    </Link>
  </div>
</article>

          <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link href={selectedHref("/admin/tournaments/status", tournamentId, tournamentName, routeDrawId)}>Status & recovery</Link>
            {String(detail.tournament.status).toLowerCase() === "draft" ? (
              <Link href={selectedHref("/admin/tournaments/delete-draft", tournamentId, tournamentName, routeDrawId)} style={{ color: "#b91c1c" }}>
                Delete draft
              </Link>
            ) : null}
          </p>
        </>
      ) : null}
    </div>
  );
}

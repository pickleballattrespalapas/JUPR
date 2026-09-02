"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import TournamentPhaseNav, {
  type TournamentPhase
} from "@/components/TournamentPhaseNav";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  tournamentId: string;
  tournamentName: string;
  drawId?: string;
  phase: TournamentPhase;
};

type StepCard = {
  title: string;
  description: string;
  href: string;
  state: "Not started" | "In progress" | "Ready" | "Complete" | "Blocked";
  note?: string;
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
  color: "#0f172a",
  textDecoration: "none"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): string {
  return tournamentRouteHref(path, { tournamentId, tournamentName, drawId });
}

function stateStyle(state: StepCard["state"]) {
  if (state === "Complete" || state === "Ready") {
    return { color: "#166534", background: "#dcfce7", borderColor: "#bbf7d0" };
  }
  if (state === "Blocked") {
    return { color: "#991b1b", background: "#fee2e2", borderColor: "#fecaca" };
  }
  if (state === "In progress") {
    return { color: "#92400e", background: "#fef3c7", borderColor: "#fde68a" };
  }
  return { color: "#475569", background: "#f8fafc", borderColor: "#cbd5e1" };
}

function phaseTitle(phase: TournamentPhase): string {
  if (phase === "setup") return "Setup";
  if (phase === "registration") return "Registration";
  if (phase === "live") return "Live Operations";
  return "Publish";
}

function phaseDescription(phase: TournamentPhase): string {
  if (phase === "setup") {
    return "Build the tournament in a clear sequence, review conflicts, and open registration only when setup is ready.";
  }
  if (phase === "registration") {
    return "Manage registrants, partners and teams, offline payments, extras, communications, and reports.";
  }
  if (phase === "live") {
    return "Prepare the tournament day, check players in, run draws and courts, score matches, and resolve corrections.";
  }
  return "Review completed results, publish divisions deliberately, and finish tournament closeout.";
}

function setupSteps(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): StepCard[] {
  const datesReady = Boolean(
    detail.tournament.start_date && detail.tournament.end_date
  );
  const policiesReady = Boolean(
    detail.settings?.registration_open_at &&
      detail.settings?.registration_close_at &&
      detail.settings?.rules_markdown &&
      detail.settings?.refund_policy_markdown &&
      detail.settings?.weather_policy_markdown
  );
  const basicsReady = Boolean(
    datesReady &&
      detail.settings?.location_name &&
      detail.settings?.timezone &&
      policiesReady
  );
  const daysReady = detail.days.length > 0;
  const eventFamilies = new Set(
    detail.event_options
      .map((event) => String(event.event_family_label || "").trim())
      .filter(Boolean)
  );
  const eventsReady = eventFamilies.size > 0;
  const divisionsReady = detail.event_options.length > 0;
  const reviewReady = basicsReady && daysReady && eventsReady && divisionsReady;
  return [
    {
      title: "1. Tournament basics and policies",
      description:
        "Name, dates, venue, timezone, sponsors, registration window, rules, cancellation policy, and weather policy.",
      href: selectedHref(
        "/admin/tournaments/setup/basics",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: basicsReady ? "Complete" : "In progress"
    },
    {
      title: "2. Schedule and courts",
      description:
        "Create the tournament days first so events and divisions can use one or several days.",
      href: selectedHref(
        "/admin/tournaments/setup/schedule",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: daysReady ? "Complete" : "Not started"
    },
    {
      title: "3. Events",
      description:
        "Create event families, choose every available day, and set draw, scoring, capacity, and pricing defaults.",
      href: selectedHref(
        "/admin/tournaments/setup/events",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: eventsReady ? "Complete" : "Not started",
      note: eventsReady
        ? `${eventFamilies.size} event${eventFamilies.size === 1 ? "" : "s"}`
        : undefined
    },
    {
      title: "4. Divisions",
      description:
        "Create skill and age divisions within each event and inherit all event days or choose a subset.",
      href: selectedHref(
        "/admin/tournaments/setup/divisions",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: divisionsReady ? "Complete" : "Not started",
      note: divisionsReady
        ? `${detail.event_options.length} division${detail.event_options.length === 1 ? "" : "s"}`
        : undefined
    },
    {
      title: "5. Pricing, extras, and fulfillment",
      description:
        "Entry fees, additional events, extras, bundles, inventory, pickup, and offline payment.",
      href: selectedHref(
        "/admin/tournaments/setup/pricing",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: "In progress"
    },
    {
      title: "6. Review and open registration",
      description:
        "Resolve missing fields, conflicts, capacity, pricing, policies, and schedule warnings before opening.",
      href: selectedHref(
        "/admin/tournaments/setup/review",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: reviewReady ? "Ready" : "Blocked",
      note: reviewReady
        ? "Basics, policies, days, events, and divisions are present."
        : "Complete basics and policies, tournament days, events, and divisions first."
    }
  ];
}


function registrationSteps(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): StepCard[] {
  const registrations = detail.summary.registrations || 0;
  const unpaid = Object.entries(detail.summary.by_payment_status || {}).reduce(
    (count, [status, value]) =>
      status.toLowerCase() === "unpaid" ? count + Number(value || 0) : count,
    0
  );
  return [
    {
      title: "Registrants",
      description: "Review identity, events, financial totals, extras, and dedicated registration edits.",
      href: selectedHref(
        "/admin/tournaments/registration/registrants",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: registrations ? "In progress" : "Not started",
      note: `${registrations} registration${registrations === 1 ? "" : "s"}`
    },
    {
      title: "Partners and teams",
      description: "Partner Board requests, automatic pairing, team rosters, substitutes, and incomplete entries.",
      href: selectedHref(
        "/admin/tournaments/registration/partners",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: registrations ? "In progress" : "Not started"
    },
    {
      title: "Payments and extras",
      description: "Record offline payments, waivers, refunds, catalog choices, pickup, and fulfillment.",
      href: selectedHref(
        "/admin/tournaments/commerce",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: unpaid ? "In progress" : registrations ? "Ready" : "Not started",
      note: unpaid ? `${unpaid} unpaid registration${unpaid === 1 ? "" : "s"}` : undefined
    },
    {
      title: "Communications and reports",
      description: "Recipient previews, dry-run email handoff, registration exports, payment reports, and check-in lists.",
      href: selectedHref(
        "/admin/tournaments/registrations",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: registrations ? "Ready" : "Not started"
    }
  ];
}

function liveSteps(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): StepCard[] {
  const registrations = detail.summary.registrations || 0;
  const coreReady = registrations > 0 && detail.event_options.length > 0;
  return [
    {
      title: "Preflight and check-in",
      description: "Registration close, partner/team completion, payments, staff, and player check-in.",
      href: selectedHref(
        "/admin/tournaments/live-operations/check-in",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: coreReady ? "In progress" : "Blocked"
    },
    {
      title: "Draws and scheduling",
      description: "Generate or import draws, assign courts and times, review conflicts, and print schedules.",
      href: selectedHref(
        "/admin/tournaments/ops/draws",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: coreReady ? "Ready" : "Blocked"
    },
    {
      title: "Live scoring",
      description: "Court assignments, score entry, progression, standings, brackets, and public live display.",
      href: selectedHref(
        "/admin/tournament-live",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: coreReady ? "In progress" : "Blocked"
    },
    {
      title: "Corrections and recovery",
      description: "Correct scores, replace players, record forfeits, and inspect recoverable operations.",
      href: selectedHref(
        "/admin/tournaments/status",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: "Ready"
    },
    {
      title: "Podium draft",
      description: "Review preliminary placements and unresolved-match warnings before publication.",
      href: selectedHref(
        "/admin/tournaments/ops",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: "Not started"
    }
  ];
}

function publishSteps(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): StepCard[] {
  const coreReady =
    detail.summary.registrations > 0 && detail.event_options.length > 0;
  return [
    {
      title: "Review results",
      description: "Missing scores, duplicates, corrections, podiums, rating paths, and replay requirements.",
      href: selectedHref(
        "/admin/tournaments/ops/results",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: coreReady ? "In progress" : "Blocked"
    },
    {
      title: "Publish divisions",
      description: "Publish each ready division, create official matches, complete replay, and expose public results.",
      href: selectedHref(
        "/admin/tournaments/ops/publish",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: coreReady ? "Not started" : "Blocked"
    },
    {
      title: "Tournament closeout",
      description: "Confirm podiums, prepare dry-run communications, finish fulfillment, close, and archive.",
      href: selectedHref(
        "/admin/tournaments/publish/closeout",
        tournamentId,
        tournamentName,
        drawId
      ),
      state: "Not started"
    }
  ];
}

export default function TournamentLifecycleOverviewPanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  tournamentName,
  drawId = "",
  phase
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    () => {
      setDetail(null);
      setBusy(false);
      setMessage(null);
    }
  );

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      if (!apiBase) throw new Error("API base URL is not configured.");
      const response = await fetch(
        `${apiUrl(apiBase, "")}/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
        { headers: { Authorization: `Bearer ${accessToken}` } }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload as AdminTournamentDetailResponse);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to load tournament."
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

  const steps = detail
    ? phase === "setup"
      ? setupSteps(detail, tournamentId, tournamentName, drawId)
      : phase === "registration"
        ? registrationSteps(detail, tournamentId, tournamentName, drawId)
        : phase === "live"
          ? liveSteps(detail, tournamentId, tournamentName, drawId)
          : publishSteps(detail, tournamentId, tournamentName, drawId)
    : [];

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <TournamentPhaseNav phase={phase} />
      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <h2 style={{ marginTop: 0 }}>{phaseTitle(phase)}</h2>
        <p style={{ color: "#334155", marginBottom: 0 }}>
          {phaseDescription(phase)}
        </p>
      </article>

      {message ? (
        <p role="alert" style={{ color: "#b91c1c" }}>{message}</p>
      ) : null}
      {busy && !detail ? <p role="status">Loading {tournamentName}…</p> : null}

      {detail ? (
        <>
          <article style={cardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
              <div>
                <strong>{detail.tournament.name}</strong>
                <div style={{ color: "#64748b" }}>
                  {detail.tournament.start_date || "Date not set"} – {detail.tournament.end_date || "Date not set"}
                </div>
              </div>
              <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                <span><strong>{detail.summary.registrations}</strong> registrations</span>
                <span><strong>{detail.event_options.length}</strong> events</span>
                <span><strong>{detail.days.length}</strong> days</span>
              </div>
            </div>
          </article>

          <section aria-label={`${phaseTitle(phase)} workflow`}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.85rem" }}>
              {steps.map((step) => (
                <Link key={step.title} href={step.href} style={phaseCardStyle}>
                  <span
                    style={{
                      width: "fit-content",
                      border: "1px solid",
                      borderRadius: "999px",
                      padding: "0.15rem 0.5rem",
                      fontSize: "0.78rem",
                      fontWeight: 800,
                      ...stateStyle(step.state)
                    }}
                  >
                    {step.state}
                  </span>
                  <strong>{step.title}</strong>
                  <span style={{ color: "#475569" }}>{step.description}</span>
                  {step.note ? <small style={{ color: "#64748b" }}>{step.note}</small> : null}
                </Link>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

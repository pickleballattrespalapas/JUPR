"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard
} from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
  tournamentId: string;
  initialName?: string | null;
};

type TournamentEdit = {
  name: string;
  startDate: string;
  endDate: string;
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
const inputStyle = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box" as const,
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
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
  tournamentName: string
): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `${path}?${params.toString()}`;
}

function editFromDetail(
  detail: AdminTournamentDetailResponse | null,
  fallbackName: string
): TournamentEdit {
  return {
    name: detail?.tournament.name || fallbackName,
    startDate: dateValue(detail?.tournament.start_date),
    endDate: dateValue(detail?.tournament.end_date)
  };
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
  tournamentName: string
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

  return [
    {
      title: "Setup",
      state: setupReady ? "Ready" : "In progress",
      description:
        "Basics, events and formats, registration rules, pricing, extras, schedule, courts, and final review.",
      href: selectedHref(
        "/admin/tournaments/setup",
        tournamentId,
        tournamentName
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
        tournamentName
      ),
      note: `${registrations} registration${registrations === 1 ? "" : "s"}`
    },
    {
      title: "Live Operations",
      state: setupReady && registrations ? "Ready" : "Blocked",
      description:
        "Preflight, check-in, draws, court schedule, live scoring, corrections, recovery, and podium draft.",
      href: selectedHref(
        "/admin/tournaments/live-operations",
        tournamentId,
        tournamentName
      ),
      note:
        setupReady && registrations
          ? "Core setup and registrations are present"
          : "Complete setup and add registrations first"
    },
    {
      title: "Publish",
      state: "Not started",
      description:
        "Review results, publish ready divisions, create official matches, complete replay, and close the tournament.",
      href: selectedHref(
        "/admin/tournaments/publish",
        tournamentId,
        tournamentName
      ),
      note: "Results become official only through Publish"
    }
  ];
}

function nextAction(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string
): { label: string; href: string; reason: string } {
  if (!detail.tournament.start_date || !detail.tournament.end_date) {
    return {
      label: "Finish tournament basics",
      href: selectedHref(
        "/admin/tournaments/setup/basics",
        tournamentId,
        tournamentName
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
        tournamentName
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
        tournamentName
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
        tournamentName
      ),
      reason: "Registration is open and no entrants are recorded yet."
    };
  }
  return {
    label: "Prepare Live Operations",
    href: selectedHref(
      "/admin/tournaments/live-operations",
      tournamentId,
      tournamentName
    ),
    reason: "Core setup and registrations are present."
  };
}

export default function TournamentHomePanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  initialName
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [edit, setEdit] = useState<TournamentEdit>(() =>
    editFromDetail(null, initialName || "")
  );
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    clearProtectedState
  );
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setEdit(editFromDetail(null, initialName || ""));
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
      const payload = await requestJson<AdminTournamentDetailResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setEdit(editFromDetail(payload, initialName || ""));
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

  async function saveTournament(confirmationText: string) {
    if (!detail?.tournament.updated_at) {
      setMessage("Reload this tournament before saving changes.");
      return;
    }
    if (edit.startDate && edit.endDate && edit.endDate < edit.startDate) {
      setMessage("Tournament end date cannot be before its start date.");
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            name: edit.name.trim(),
            start_date: edit.startDate || null,
            end_date: edit.endDate || null,
            expected_updated_at: detail.tournament.updated_at,
            confirmation_text: confirmationText,
            source: "next_tournament_lifecycle_home"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(
        payload.idempotent_replay
          ? "Tournament update safely reconciled."
          : "Tournament details saved."
      );
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to save tournament details."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
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
  const next = detail ? nextAction(detail, tournamentId, tournamentName) : null;
  const phases = detail
    ? phaseCards(detail, tournamentId, tournamentName)
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
      href={selectedHref("/admin/tournaments/setup/basics", tournamentId, tournamentName)}
      style={{ display: "inline-block", padding: "0.6rem 0.9rem", borderRadius: "999px", background: "#0f172a", color: "white", textDecoration: "none", fontWeight: 800 }}
    >
      Edit in guided setup
    </Link>
  </div>
</article>

          <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link href={selectedHref("/admin/tournaments/status", tournamentId, tournamentName)}>Status & recovery</Link>
            {String(detail.tournament.status).toLowerCase() === "draft" ? (
              <Link href={selectedHref("/admin/tournaments/delete-draft", tournamentId, tournamentName)} style={{ color: "#b91c1c" }}>
                Delete draft
              </Link>
            ) : null}
          </p>
        </>
      ) : null}
    </div>
  );
}

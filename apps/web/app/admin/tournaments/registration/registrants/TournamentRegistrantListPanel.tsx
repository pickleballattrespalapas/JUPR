"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type {
  AdminTournamentDetailResponse,
  AdminTournamentRegistration,
  AdminTournamentStatusResponse
} from "@/lib/adminTournamentApi";
import {
  formatCommerceMoney,
  getAdminTournamentCommerceDetail,
  type AdminTournamentCommerceDetail
} from "@/lib/tournamentCommerceApi";
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
  tournamentName: string;
  drawId: string;
};

type RegistrationSummary = {
  registration: AdminTournamentRegistration;
  events: string[];
  extras: string[];
  totalMinor: number | null;
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

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function recordValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): Record<string, unknown> {
  const value = row?.[key];
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function recordList(
  row: Record<string, unknown> | null | undefined,
  key: string
): Array<Record<string, unknown>> {
  const value = row?.[key];
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is Record<string, unknown> =>
          Boolean(entry) && typeof entry === "object"
      )
    : [];
}

function stringValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): string {
  const value = row?.[key];
  return value == null ? "" : String(value);
}

function numberValue(
  row: Record<string, unknown> | null | undefined,
  key: string
): number | null {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : null;
}

function selectedHref(
  path: string,
  tournamentId: string,
  tournamentName: string,
  drawId: string
): string {
  return tournamentRouteHref(path, { tournamentId, tournamentName, drawId });
}

function orderForRegistration(
  commerce: AdminTournamentCommerceDetail | null,
  registrationId: string
): Record<string, unknown> | null {
  return (
    commerce?.orders.find(
      (order) => stringValue(order, "registration_id") === registrationId
    ) || null
  );
}

function extraLabels(order: Record<string, unknown> | null): string[] {
  const quote = recordValue(order, "quote");
  return recordList(quote, "lines")
    .filter((line) =>
      ["ITEM", "BUNDLE"].includes(stringValue(line, "line_type").toUpperCase())
    )
    .map((line) => {
      const label = stringValue(line, "label") || "Extra";
      const option = stringValue(line, "option_label");
      const quantity = numberValue(line, "quantity") || 1;
      return `${quantity > 1 ? `${quantity}× ` : ""}${label}${option ? ` — ${option}` : ""}`;
    });
}

function orderTotal(order: Record<string, unknown> | null): number | null {
  const direct = numberValue(order, "total_minor");
  if (direct != null) return direct;
  return numberValue(recordValue(order, "quote"), "total_minor");
}

export default function TournamentRegistrantListPanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  tournamentName,
  drawId
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [commerce, setCommerce] =
    useState<AdminTournamentCommerceDetail | null>(null);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const request = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    () => {
      setDetail(null);
      setCommerce(null);
      setSearch("");
      setBusy(false);
      setMessage(null);
    }
  );

  async function loadWorkspace() {
    const generation = request.begin();
    setBusy(true);
    setMessage(null);
    try {
      if (!apiBase) throw new Error("API base URL is not configured.");
      const [detailResponse, commerceResponse] = await Promise.all([
        fetch(
          `${apiUrl(apiBase, "")}/admin/clubs/${encodeURIComponent(
            clubId
          )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
          { headers: { Authorization: `Bearer ${accessToken}` } }
        ),
        getAdminTournamentCommerceDetail(clubId, tournamentId, accessToken)
      ]);
      const detailPayload = await detailResponse.json().catch(() => null);
      if (!detailResponse.ok) {
        throw new Error(
          String(
            detailPayload?.detail || `API error (${detailResponse.status})`
          )
        );
      }
      if (!request.isCurrent(generation)) return;
      setDetail(detailPayload as AdminTournamentDetailResponse);
      setCommerce(commerceResponse.data || null);
      if (commerceResponse.error) {
        setMessage(
          "Registration data loaded. Financial and extras summaries are temporarily unavailable."
        );
      }
    } catch (error) {
      if (request.isCurrent(generation)) {
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to load registrations."
        );
      }
    } finally {
      if (request.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status.enabled ? `${accessToken}\u0000${tournamentId}` : "",
    loadWorkspace
  );

  const summaries = useMemo<RegistrationSummary[]>(() => {
    if (!detail) return [];
    return detail.registrations.map((registration) => {
      const events = detail.selections
        .filter((selection) => selection.registration_id === registration.id)
        .map(
          (selection) =>
            selection.event_label || selection.event_option_id || "Event"
        );
      const order = orderForRegistration(commerce, registration.id);
      return {
        registration,
        events,
        extras: extraLabels(order),
        totalMinor: orderTotal(order)
      };
    });
  }, [commerce, detail]);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return summaries;
    return summaries.filter(({ registration, events, extras }) =>
      [
        registration.display_name,
        registration.email,
        registration.phone,
        ...events,
        ...extras
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [search, summaries]);

  if (sessionLoading && !accessToken) {
    return <p role="status">Loading registration workspace…</p>;
  }
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {message ? (
        <p
          role="status"
          style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : "#92400e" }}
        >
          {message}
        </p>
      ) : null}
      <article style={cardStyle}>
        <label>
          <strong>Search registrations</strong><br />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Name, email, phone, event, or extra"
            style={inputStyle}
          />
        </label>
      </article>

      {busy && !detail ? <p role="status">Loading {tournamentName} registrations…</p> : null}

      {detail ? (
        <section aria-label="Tournament registrations" style={{ display: "grid", gap: "0.75rem" }}>
          {visible.map(({ registration, events, extras, totalMinor }) => (
            <article key={registration.id} style={cardStyle}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
                <div style={{ minWidth: 0 }}>
                  <h2 style={{ margin: 0 }}>{registration.display_name}</h2>
                  <p style={{ color: "#475569", margin: "0.25rem 0 0", overflowWrap: "anywhere" }}>
                    {registration.email || "No email"}
                    {registration.phone ? ` · ${registration.phone}` : ""}
                  </p>
                </div>
                <Link
                  href={selectedHref(
                    `/admin/tournaments/registration/registrants/${encodeURIComponent(
                      registration.id
                    )}`,
                    tournamentId,
                    tournamentName,
                    drawId
                  )}
                  style={{ display: "inline-block", padding: "0.5rem 0.8rem", borderRadius: "999px", border: "1px solid #0f172a", textDecoration: "none", color: "#0f172a", fontWeight: 800 }}
                >
                  Edit registration
                </Link>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
                <div><strong>Status</strong><br />{registration.registration_status || "—"}</div>
                <div><strong>Payment</strong><br />{registration.payment_status || "—"}</div>
                <div><strong>Amount</strong><br />{totalMinor == null ? "—" : formatCommerceMoney(totalMinor)}</div>
                <div><strong>Events</strong><br />{events.length ? events.join(", ") : "None"}</div>
                <div><strong>Extras</strong><br />{extras.length ? extras.join(", ") : "None"}</div>
              </div>
            </article>
          ))}
          {!visible.length ? (
            <article style={cardStyle}><p style={{ margin: 0, color: "#64748b" }}>No registrations match this search.</p></article>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}

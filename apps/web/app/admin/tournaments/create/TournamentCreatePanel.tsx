"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminTournamentStatusResponse } from "@/lib/adminTournamentApi";
import { useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
};

type CreateCommand = {
  clubId: string;
  tournamentId: string;
  idempotencyKey: string;
  name: string;
  startDate: string;
  endDate: string;
};

type WriteResponse = {
  ok: boolean;
  tournament?: Record<string, unknown> | null;
  idempotent_replay?: boolean;
};

const storageKey = "jupr_tournament_manager_create_command_v1";
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function readStoredCommand(clubId: string): CreateCommand | null {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) || "null") as Partial<CreateCommand> | null;
    if (!parsed || parsed.clubId !== clubId || !parsed.tournamentId || !parsed.idempotencyKey || !parsed.name) return null;
    return {
      clubId,
      tournamentId: String(parsed.tournamentId),
      idempotencyKey: String(parsed.idempotencyKey),
      name: String(parsed.name),
      startDate: String(parsed.startDate || ""),
      endDate: String(parsed.endDate || "")
    };
  } catch {
    return null;
  }
}

function persistCommand(command: CreateCommand | null) {
  try {
    if (command) window.localStorage.setItem(storageKey, JSON.stringify(command));
    else window.localStorage.removeItem(storageKey);
  } catch {
    // The current-page retry remains available when storage is blocked.
  }
}

function tournamentHomeHref(tournamentId: string, name: string): string {
  const params = new URLSearchParams({ tournament: tournamentId, name });
  return `/admin/tournaments/tournament?${params.toString()}`;
}

export default function TournamentCreatePanel({ apiBase, clubId, status }: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const actionRequest = useLatestRequestGuard(accessToken);
  const [name, setName] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [command, setCommand] = useState<CreateCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const stored = readStoredCommand(clubId);
    if (!stored) return;
    setCommand(stored);
    setName(stored.name);
    setStartDate(stored.startDate);
    setEndDate(stored.endDate);
  }, [clubId]);

  function updateField(setter: (value: string) => void, value: string) {
    setter(value);
    setCommand(null);
    persistCommand(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before creating a tournament.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  async function createTournament(confirmationText: string) {
    const cleanName = name.trim();
    if (!cleanName) {
      setMessage("Tournament name is required.");
      return;
    }
    if (startDate && endDate && endDate < startDate) {
      setMessage("Tournament end date cannot be before its start date.");
      return;
    }

    const requestCommand = command || {
      clubId,
      tournamentId: globalThis.crypto.randomUUID(),
      idempotencyKey: globalThis.crypto.randomUUID(),
      name: cleanName,
      startDate,
      endDate
    };
    if (!command) {
      setCommand(requestCommand);
      persistCommand(requestCommand);
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments`,
        {
          method: "POST",
          body: JSON.stringify({
            tournament_id: requestCommand.tournamentId,
            idempotency_key: requestCommand.idempotencyKey,
            name: requestCommand.name,
            start_date: requestCommand.startDate || null,
            end_date: requestCommand.endDate || null,
            confirmation_text: confirmationText
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      const createdId = String(payload.tournament?.id || requestCommand.tournamentId);
      const createdName = String(payload.tournament?.name || requestCommand.name);
      setCommand(null);
      persistCommand(null);
      router.push(tournamentHomeHref(createdId, createdName));
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(`${error instanceof Error ? error.message : "Unable to create tournament."} Retry keeps the same protected request.`);
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}>Tournament creation is currently unavailable.</article>;
  if (sessionLoading) return <p role="status">Checking admin access…</p>;
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  const ready = Boolean(name.trim() && (!startDate || !endDate || endDate >= startDate));
  return (
    <article style={cardStyle}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
        <label><strong>Tournament name</strong><br /><input value={name} onChange={(event) => updateField(setName, event.target.value)} maxLength={180} style={inputStyle} /></label>
        <label><strong>Start date</strong><br /><input type="date" value={startDate} onChange={(event) => updateField(setStartDate, event.target.value)} style={inputStyle} /></label>
        <label><strong>End date</strong><br /><input type="date" value={endDate} min={startDate || undefined} onChange={(event) => updateField(setEndDate, event.target.value)} style={inputStyle} /></label>
      </div>
      {startDate && endDate && endDate < startDate ? <p role="alert" style={{ color: "#b91c1c" }}>End date must be on or after the start date.</p> : null}
      <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
        <ConfirmAction
          triggerLabel={busy ? "Creating…" : "Create tournament"}
          title="Create this tournament draft?"
          description="This creates one draft tournament. Registration remains closed until setup is reviewed and published."
          confirmLabel="Yes, create tournament"
          confirmationText="CREATE TOURNAMENT"
          disabled={!ready}
          busy={busy}
          onConfirm={createTournament}
        />
        <Link href="/admin/tournaments">Cancel</Link>
      </p>
      {command ? <p style={{ color: "#92400e" }}>A protected retry is ready. Keep the form unchanged and submit the same request again if the first response was interrupted.</p> : null}
      {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}

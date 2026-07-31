"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerSchedulePreviewResponse,
  AdminLeagueManagerStatusResponse,
  AdminLeagueManagerWriteResponse
} from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import { GuidedLeagueSettingsEditor } from "../GuidedLeagueSettingsEditor";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminLeagueManagerStatusResponse;
  initialLeague: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function LeagueSettingsPanel({ apiBase, clubId, status, initialLeague }: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const detailRequest = useLatestRequestGuard(`${accessToken}\u0000${initialLeague}`, clearProtectedState);
  const actionRequest = useLatestRequestGuard(accessToken);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before editing league settings.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setBusy(false);
    setMessage(null);
  }

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load league settings.");
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveSettings(patch: Record<string, unknown>, confirmationText: string): Promise<boolean> {
    if (!detail) return false;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            ...patch,
            confirmation_text: confirmationText,
            source: "next_selected_league_settings_page"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return false;
      if (payload.detail) setDetail(payload.detail);
      else await loadDetail();
      setMessage(`Saved settings for ${initialLeague}.`);
      return true;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save league settings.");
      }
      return false;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function previewSchedule(scheduleConfig: Record<string, unknown>): Promise<AdminLeagueManagerSchedulePreviewResponse | null> {
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerSchedulePreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(initialLeague)}/schedule/preview`,
        { method: "POST", body: JSON.stringify({ schedule_config: scheduleConfig }) }
      );
      return actionRequest.isCurrent(generation) ? payload : null;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to preview the schedule.");
      }
      return null;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\u0000${initialLeague}` : "", loadDetail);

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}>League Manager is currently unavailable.</article>;
  }

  if (sessionLoading) return <p role="status">Checking admin access…</p>;

  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      {busy && !detail ? <p role="status">Loading {initialLeague} settings…</p> : null}
      {message ? <p role="status" style={{ color: /unable|error|required/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      {detail ? (
        <GuidedLeagueSettingsEditor
          detail={detail}
          saving={busy}
          canWrite={Boolean(accessToken)}
          onSave={saveSettings}
          onPreview={previewSchedule}
        />
      ) : null}
    </div>
  );
}

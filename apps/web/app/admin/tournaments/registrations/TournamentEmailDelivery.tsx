"use client";

import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { InteractionActionError, type ActionCompletion } from "@/components/interaction/types";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import type { AdminTournamentBroadcastPreviewResponse, TournamentBroadcast, TournamentBroadcastSummary } from "@/lib/adminTournamentApi";

type Props = {
  clubId: string;
  tournamentId: string;
  accessToken: string;
  apiBase: string | null;
  preview: AdminTournamentBroadcastPreviewResponse | null;
  previewScope: string;
  subject: string;
  message: string;
  includeCancelled: boolean;
  includeRegistrationEvents?: boolean;
  busy: boolean;
  onBusy: (value: boolean) => void;
  requestJson: <T>(path: string, options?: RequestInit) => Promise<T>;
};
const CONFIRM_SEND = "SEND TO SELECTED PARTICIPANTS";
const buttonStyle = { border: "1px solid #cbd5e1", borderRadius: "8px", background: "white", padding: "0.6rem", minHeight: "44px" };
const labels: Record<string, string> = { pending: "Not sent", sent: "Sent", dry_run: "Test complete", staging_redirect: "Sent to test inbox", uncertain: "Check delivery" };

function outcome(result: TournamentBroadcast): string {
  const count = (status: string) => result.recipients.filter(row => row.status === status).length;
  return [count("sent") ? `${count("sent")} sent` : "", count("dry_run") ? `${count("dry_run")} tested` : "",
    count("staging_redirect") ? `${count("staging_redirect")} sent to test inbox` : "",
    count("uncertain") ? `${count("uncertain")} need a delivery check` : "",
    count("pending") ? `${count("pending")} not sent` : ""].filter(Boolean).join(" · ");
}

export default function TournamentEmailDelivery(props: Props) {
  const { clubId, tournamentId, accessToken, apiBase, preview, previewScope, subject, message, includeCancelled, includeRegistrationEvents = false, busy, onBusy, requestJson } = props;
  const [history, setHistory] = useState<TournamentBroadcastSummary[]>([]);
  const [active, setActive] = useState<TournamentBroadcast | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(true);
  const inFlight = useRef(false);
  const draftOperation = useRef<{ fingerprint: string; key: string } | null>(null);
  const currentPreviewScope = useRef(previewScope);
  currentPreviewScope.current = previewScope;
  const context = `${accessToken}\u0000${apiBase}\u0000${clubId}\u0000${tournamentId}`;
  const scope = useLatestRequestGuard(context, () => {
    setActive(null); setHistory([]); setError(""); setSending(false); onBusy(false);
    inFlight.current = false; draftOperation.current = null;
  });
  const historyScope = useLatestRequestGuard(context);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const path = `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registrations/broadcasts`;

  async function loadHistory() {
    const generation = historyScope.begin();
    try {
      const data = await requestJson<{ broadcasts: TournamentBroadcastSummary[] }>(path, { cache: "no-store" });
      if (mounted.current && historyScope.isCurrent(generation)) setHistory(data.broadcasts || []);
    } catch {
      if (mounted.current && historyScope.isCurrent(generation)) setError("Recent emails could not be loaded. Refresh their results before sending an email again.");
    }
  }
  useAuthenticatedAutoLoad(accessToken, loadHistory, `${apiBase}\u0000${clubId}\u0000${tournamentId}`);

  async function readSaved(key: string): Promise<TournamentBroadcast> {
    return requestJson<TournamentBroadcast>(`${path}/${encodeURIComponent(key)}`, { cache: "no-store" });
  }

  async function viewSaved(key: string): Promise<void> {
    const generation = scope.begin();
    setError("");
    try {
      const result = await readSaved(key);
      if (mounted.current && scope.isCurrent(generation)) setActive(result);
    } catch {
      if (mounted.current && scope.isCurrent(generation)) setError("Email results could not be loaded. Try refreshing the results.");
    }
  }

  async function send(confirmationText: string, resumeKey?: string): Promise<ActionCompletion> {
    if (inFlight.current) throw new InteractionActionError("This email is already being processed.");
    if (!resumeKey && (!preview?.send_available || !preview.preview_fingerprint || currentPreviewScope.current !== previewScope)) {
      throw new InteractionActionError("Your email changed. Review a fresh preview before sending.");
    }
    const fingerprint = preview?.preview_fingerprint || "";
    if (!resumeKey && draftOperation.current?.fingerprint !== fingerprint) {
      draftOperation.current = { fingerprint, key: crypto.randomUUID() };
    }
    const key = resumeKey || draftOperation.current!.key;
    const generation = scope.begin();
    const isCurrent = () => mounted.current && scope.isCurrent(generation);
    inFlight.current = true;
    setSending(true); onBusy(true); setError("");
    try {
      let result = resumeKey ? await readSaved(key) : await requestJson<TournamentBroadcast>(path, {
        method: "POST", body: JSON.stringify({ operation_key: key, preview_fingerprint: fingerprint,
          registration_ids: preview!.selected_registration_ids, subject, message,
          include_cancelled: includeCancelled, include_registration_events: includeRegistrationEvents, confirmation_text: confirmationText })
      });
      if (!isCurrent()) throw new Error("The admin session changed.");
      setActive(result);
      for (const recipient of result.recipients.filter(row => row.status === "pending")) {
        if (!isCurrent()) throw new Error("The admin session changed.");
        const delivered = await requestJson<{ index: number; status: string; detail: string }>(
          `${path}/${encodeURIComponent(key)}/recipients/${recipient.index}/send`, {
            method: "POST", body: JSON.stringify({ confirmation_text: confirmationText })
          });
        result = { ...result, recipients: result.recipients.map(row => row.index === recipient.index ? { ...row, ...delivered } : row) };
        result.pending_count = result.recipients.filter(row => row.status === "pending").length;
        if (isCurrent()) setActive(result);
      }
      if (isCurrent()) await loadHistory();
      return { status: "success", title: result.delivery_mode === "live" ? "Email results" : "Email test results", description: outcome(result) };
    } catch (caught) {
      const status = (caught as { status?: number })?.status;
      if (status && status >= 400 && status < 500) {
        const explanation = caught instanceof Error ? caught.message : "The email could not be sent. Review your selection.";
        if (isCurrent()) { setError(explanation); await loadHistory(); }
        throw new InteractionActionError(explanation, { kind: status === 403 ? "forbidden" : "validation" });
      }
      if (isCurrent()) {
        setError("Sending paused. Check the saved results below; you can continue with participants who have not been sent an email.");
        await loadHistory();
      }
      return { status: "uncertain", title: "Check email results", operationKey: key, showOperationReference: false,
        description: "Sending paused. Check the saved results before continuing. Emails already attempted will not be sent again.",
        recoveryLabel: "Check saved results", onRecover: async () => {
          if (!isCurrent()) throw new InteractionActionError("Reopen Recent emails in the current admin session.");
          const result = await readSaved(key);
          if (isCurrent()) { setActive(result); setError(""); }
          return { status: "success", title: "Saved email results", description: outcome(result) };
        } };
    } finally {
      if (isCurrent()) { setSending(false); onBusy(false); inFlight.current = false; }
    }
  }

  const alreadyHandled = !!active && draftOperation.current?.key === active.operation_key && draftOperation.current?.fingerprint === preview?.preview_fingerprint;
  return <div style={{ marginTop: "1rem" }}>
    {preview ? <>
      <p><strong>From:</strong> {preview.sender?.from_name} {preview.sender?.from_email}
        {preview.sender?.reply_to ? <><br /><strong>Replies to:</strong> {preview.sender.reply_to}</> : null}</p>
      {preview.delivery_mode !== "live" ? <p role="status">Test mode: participants will not receive this email.</p> : null}
      <ConfirmAction
        triggerLabel={preview.delivery_mode === "live" ? `Send email to ${preview.recipient_count} recipient${preview.recipient_count === 1 ? "" : "s"}` : "Test email sending"}
        title={preview.delivery_mode === "live" ? "Send this email?" : "Test this email?"}
        description={`Send “${preview.preview.subject}” to ${preview.recipient_count} selected email recipient${preview.recipient_count === 1 ? "" : "s"}. Each person receives a separate email.${includeRegistrationEvents ? " Each email includes that recipient’s registration events." : ""}`}
        preview={<ul>{preview.recipients.map(row => <li key={row.email}>{row.name} · {row.email}</li>)}</ul>}
        confirmLabel={preview.delivery_mode === "live" ? "Yes, send email" : "Yes, run test"}
        workingLabel="Sending email…" confirmationText={CONFIRM_SEND}
        disabled={busy || !preview.send_available || !preview.preview_fingerprint || alreadyHandled}
        disabledReason={!preview.send_available ? preview.send_unavailable_reason : alreadyHandled ? "This email is recorded below. Use its saved results to continue." : undefined}
        onConfirm={send} />
    </> : null}
    {active ? <section aria-label="Email results" style={{ marginTop: "1rem", padding: "0.75rem", background: "#f8fafc", borderRadius: "10px" }}>
      <h3 style={{ marginTop: 0 }}>{active.subject}</h3>
      <p role="status" aria-live="polite">{sending ? "Sending… " : ""}{outcome(active)}</p>
      <details><summary>View saved message</summary><p style={{ whiteSpace: "pre-wrap" }}>{active.message}</p>{active.include_registration_events ? <p>Each email includes that recipient’s registration events.</p> : null}</details>
      <ul style={{ paddingLeft: "1.25rem" }}>{active.recipients.map(row => <li key={row.email} style={{ marginTop: "0.5rem", overflowWrap: "anywhere" }}>
        <strong>{row.name}</strong> · {row.email} · <strong>{labels[row.status] || "Check delivery"}</strong>
        {row.status === "uncertain" ? <p>{row.detail}</p> : null}
      </li>)}</ul>
      <p style={{ color: "#475569" }}>“Sent” means the mail server accepted the email. Check the recipient’s inbox to confirm arrival.</p>
      <button type="button" disabled={busy} style={buttonStyle} onClick={() => viewSaved(active.operation_key)}>Refresh email results</button>{" "}
      {active.pending_count > 0 ? <ConfirmAction triggerLabel={`Continue sending (${active.pending_count} remaining)`}
        title="Send the remaining emails?" description={`Continue “${active.subject}” for ${active.pending_count} recipients who have not been sent this email.`}
        preview={<ul>{active.recipients.filter(row => row.status === "pending").map(row => <li key={row.email}>{row.name} · {row.email}</li>)}</ul>}
        confirmLabel="Yes, continue sending" confirmationText={CONFIRM_SEND} disabled={busy}
        onConfirm={text => send(text, active.operation_key)} /> : null}
    </section> : null}
    {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
    <details style={{ marginTop: "1rem" }}>
      <summary>Recent emails</summary>
      <p><button type="button" disabled={busy} style={buttonStyle} onClick={loadHistory}>Refresh recent emails</button></p>
      {history.length ? <ul>{history.map(row => <li key={row.operation_key} style={{ marginTop: "0.5rem" }}>
        <button type="button" disabled={busy} style={buttonStyle} onClick={() => viewSaved(row.operation_key)}>{row.subject} · {row.recipient_count} recipient{row.recipient_count === 1 ? "" : "s"}</button>
      </li>)}</ul> : <p>No recorded emails yet.</p>}
    </details>
  </div>;
}

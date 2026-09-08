"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { ADMIN_WORKSPACE_DETAILS_CHANGE } from "@/lib/adminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import styles from "./settings.module.css";

type Club = {
  id: string; slug: string; name: string; tagline: string | null;
  support_email: string | null; is_active: boolean;
  onboarding_status: string; updated_at: string;
};
type Settings = { club: Club; setup: { missing: string[]; can_submit: boolean } };
const statusLabels: Record<string, string> = {
  draft: "Getting started", in_progress: "Setup in progress",
  ready_for_review: "Submitted for review", ready: "Setup complete", completed: "Setup complete"
};

export default function ClubSettingsPage() {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const canManage = session?.capabilities?.assignments.some(a =>
    a.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(a.role));
  if (loading) return <p role="status">Checking club access…</p>;
  if (!accessToken || !canManage) return <p>Club administrator access is required to edit club settings.</p>;
  // Changing account or club unmounts the editor, including drafts and requests.
  return <ClubSettingsEditor key={`${clubId}:${session?.user?.id || session?.user?.email || accessToken}`} clubId={clubId} accessToken={accessToken}/>;
}

function ClubSettingsEditor({ clubId, accessToken }: { clubId: string; accessToken: string }) {
  const api = getAdminApiBaseUrl();
  const token = useRef(accessToken);
  token.current = accessToken;
  const [settings, setSettings] = useState<Settings | null>(null);
  const [name, setName] = useState("");
  const [tagline, setTagline] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [needsReload, setNeedsReload] = useState(false);
  const [revision, setRevision] = useState(0);
  const saving = useRef(false);
  const pendingSave = useRef<AbortController | null>(null);

  function accept(data: Settings) {
    setSettings(data); setName(data.club.name); setTagline(data.club.tagline || ""); setEmail(data.club.support_email || "");
  }

  useEffect(() => {
    const controller = new AbortController();
    setSettings(null); setError(""); setMessage(""); setNeedsReload(false);
    async function load() {
      try {
        if (!api) throw new Error("Club settings are temporarily unavailable.");
        const response = await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/settings`, {
          headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal
        });
        const data = await response.json();
        if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Unable to load club settings.");
        if (!controller.signal.aborted) accept(data);
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "Unable to load club settings.");
      }
    }
    void load();
    return () => controller.abort();
  }, [api, clubId, revision]);

  useEffect(() => () => pendingSave.current?.abort(), []);

  async function save(submit: boolean) {
    if (!api || !settings || saving.current || needsReload) return;
    saving.current = true; setBusy(true); setError(""); setMessage("");
    const controller = new AbortController(); pendingSave.current = controller;
    let receivedResult = false;
    try {
      const response = await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/settings`, {
        method: "PUT", headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ name, tagline, support_email: email, expected_updated_at: settings.club.updated_at, submit_for_review: submit })
      });
      const data = await response.json();
      if (controller.signal.aborted) return;
      receivedResult = true;
      if (!response.ok) {
        setNeedsReload(response.status === 409 || response.status === 401 || response.status === 403 || response.status >= 500);
        throw new Error(typeof data.detail === "string" ? data.detail : "Check the club name and contact email.");
      }
      accept(data);
      setMessage(submit ? "Your club setup has been submitted for Super Admin review." : "Club settings saved.");
      window.dispatchEvent(new Event(ADMIN_WORKSPACE_DETAILS_CHANGE));
    } catch (e) {
      if (!controller.signal.aborted) {
        if (!receivedResult) setNeedsReload(true);
        setError(receivedResult && e instanceof Error ? e.message : "Could not confirm the save. Reload before trying again.");
      }
    } finally {
      if (!controller.signal.aborted) { saving.current = false; setBusy(false); }
    }
  }

  const club = settings?.club;
  const dirty = club && (name !== club.name || tagline !== (club.tagline || "") || email !== (club.support_email || ""));
  return <section className={styles.page}>
    <h1>Club settings</h1>
    <p>Manage your club’s name, description, and contact details.</p>
    {error && <div role="alert" className={styles.error}><p>{error}</p>{(!settings || needsReload) && <button disabled={busy} onClick={() => setRevision(n => n + 1)}>Reload club settings</button>}{needsReload && dirty && <p>Reloading will replace your unsaved changes with the latest saved details.</p>}</div>}
    {message && <p role="status" className={styles.success}>{message}</p>}
    {!settings && !error && <p role="status">Loading club settings…</p>}
    {club && settings && <>
      <div className={styles.card}>
        <h2>{club.name}</h2>
        <p><strong>{club.is_active ? "Active club" : statusLabels[club.onboarding_status] || "Setup under review"}</strong></p>
        {club.onboarding_status === "ready_for_review" && !club.is_active && <p>Super Admin can review your setup now. If you change these details, submit them again for review.</p>}
        {settings.setup.can_submit && club.onboarding_status !== "ready_for_review" && <p>Add a contact email, check your details below, and submit your setup when you’re ready.</p>}
        <p>Club page: <code>/clubs/{club.slug}</code>{club.is_active && <> · <Link href={`/clubs/${encodeURIComponent(club.slug)}`}>View club page</Link></>}</p>
      </div>
      <form className={styles.card} onSubmit={e => { e.preventDefault(); void save(false); }}>
        <fieldset disabled={busy || needsReload} className={styles.fields}>
          <legend>Club details</legend>
          <label>Club name<input required maxLength={120} value={name} onChange={e => setName(e.target.value)}/></label>
          <label>Short description<textarea maxLength={240} rows={3} value={tagline} onChange={e => setTagline(e.target.value)} aria-describedby="description-help"/></label>
          <small id="description-help">A short introduction to your club. Up to 240 characters.</small>
          <label>Contact email<input type="email" maxLength={254} value={email} onChange={e => setEmail(e.target.value)} aria-describedby="contact-help"/></label>
          <small id="contact-help">Use an address players can contact. This may appear on your public club pages.</small>
          <div className={styles.actions}>
            <button type="submit" disabled={!name.trim()}>{busy ? "Saving…" : "Save details"}</button>
            {settings.setup.can_submit && <button type="button" disabled={!name.trim() || !email.trim() || (club.onboarding_status === "ready_for_review" && !dirty)} onClick={e => { if (e.currentTarget.form?.reportValidity()) void save(true); }}>Save and submit for review</button>}
          </div>
        </fieldset>
      </form>
      <div className={styles.card}>
        <h2>People and programs</h2>
        <p><Link href="/admin/staff">Manage club staff</Link> · <Link href="/admin/players">Manage players</Link> · <Link href="/admin/interclub">Plan an interclub season</Link></p>
      </div>
    </>}
  </section>;
}

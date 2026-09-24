"use client";
import { useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { leaderboardSettingsError, type LeaderboardSettings } from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";
import LeaderboardSettingsEditor from "./LeaderboardSettingsEditor";

type SettingsDocument = { leaderboard: LeaderboardSettings };
type SettingsRecord = { revision: number; draft: LeaderboardSettings; published: LeaderboardSettings | null; published_at: string | null };

export default function LeaderboardSettingsPage() {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const allowed = session?.capabilities?.assignments.some(
    (a) =>
      a.club_id === clubId &&
      ["administrator", "club_owner", "super_admin"].includes(a.role),
  );
  if (loading) return <p role="status">Checking club access…</p>;
  if (!accessToken || !allowed)
    return <p>Club administrator access is required to edit leaderboard settings.</p>;
  return (
    <SettingsEditor
      key={`${clubId}:${session?.user?.id || session?.user?.email || accessToken}`}
      clubId={clubId}
      accessToken={accessToken}
    />
  );
}

function SettingsEditor({
  clubId,
  accessToken,
}: {
  clubId: string;
  accessToken: string;
}) {
  const [site, setSite] = useState<SettingsRecord | null>(null),
    [doc, setDoc] = useState<SettingsDocument | null>(null);
  const [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0);
  const token = useRef(accessToken);
  token.current = accessToken;
  const lock = useRef(false),
    pending = useRef<AbortController | null>(null);
  const api = getAdminApiBaseUrl(),
    endpoint = `${api}/admin/clubs/${encodeURIComponent(clubId)}/leaderboard-settings`;
  useEffect(() => {
    const controller = new AbortController();
    setSite(null);
    setDoc(null);
    setError("");
    setBlocked(false);
    void (async () => {
      try {
        const response = await fetch(endpoint, {
          headers: { Authorization: `Bearer ${token.current}` },
          cache: "no-store",
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok)
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "Unable to load leaderboard settings.",
          );
        if (!controller.signal.aborted) {
          setSite(data);
          setDoc({ leaderboard: data.draft });
        }
      } catch (e) {
        if (!controller.signal.aborted)
          setError(
            e instanceof Error ? e.message : "Unable to load leaderboard settings.",
          );
      }
    })();
    return () => controller.abort();
  }, [endpoint, reload]);
  useEffect(() => () => pending.current?.abort(), []);
  const dirty =
    !!doc && !!site && JSON.stringify(doc.leaderboard) !== JSON.stringify(site.draft);
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  async function mutate(action: "save" | "publish" | "discard") {
    if (!site || !doc || lock.current || blocked) return;
    const leaderboardError = action === "save" ? leaderboardSettingsError(doc.leaderboard) : null;
    if (leaderboardError) {
      setError(leaderboardError);
      return;
    }
    if (readBrowserWorkspace()?.clubId !== clubId) {
      setError(
        "Your selected club changed in another tab. Reopen the workspace before saving.",
      );
      setBlocked(true);
      return;
    }
    if (action === "publish" && dirty) {
      setError("Save the draft before publishing.");
      return;
    }
    lock.current = true;
    setBusy(true);
    setError("");
    setMessage("");
    const controller = new AbortController();
    pending.current = controller;
    let received = false;
    try {
      const response = await fetch(
        action === "save" ? endpoint : `${endpoint}/${action}`,
        {
          method: action === "save" ? "PUT" : "POST",
          headers: {
            Authorization: `Bearer ${token.current}`,
            "Content-Type": "application/json",
          },
          signal: controller.signal,
          body: JSON.stringify({
            revision: site.revision,
            ...(action === "save" ? { settings: doc.leaderboard } : {}),
          }),
        },
      );
      const data = await response.json();
      if (controller.signal.aborted) return;
      received = true;
      if (!response.ok) {
        if ([401, 403, 409].includes(response.status) || response.status >= 500)
          setBlocked(true);
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg: string }) => d.msg).join(" ")
              : "Check your leaderboard settings.",
        );
      }
      setSite(data);
      setDoc({ leaderboard: data.draft });
      setMessage(
        action === "save"
          ? "Draft saved. Publish to apply these leaderboard settings."
          : action === "publish"
            ? "Leaderboard settings published."
              : "Draft restored to the published version.",
      );
    } catch (e) {
      if (!controller.signal.aborted) {
        if (!received) setBlocked(true);
        setError(
          e instanceof Error
            ? e.message
            : "Could not confirm the save. Reload before trying again.",
        );
      }
    } finally {
      if (!controller.signal.aborted) {
        lock.current = false;
        setBusy(false);
      }
    }
  }
  function change(patch: Partial<SettingsDocument>) {
    if (doc) {
      setDoc({ ...doc, ...patch });
      setMessage("");
    }
  }
  return <section>
    <h1>Leaderboard settings</h1>
    <p>Choose the statistics and seasons visitors see on the Overall leaderboard.</p>
    {error ? <p role="alert">{error}</p> : null}
    {message ? <p role="status">{message}</p> : null}
    {(!site || !doc) && !error ? <p role="status">Loading settings…</p> : null}
    {blocked || !site ? <button type="button" disabled={busy} onClick={() => setReload(n => n + 1)}>Reload settings</button> : null}
    {site && doc ? <>
      <div className={styles.actions}>
        <button type="button" disabled={busy || blocked || !dirty} onClick={() => void mutate("save")}>Save draft</button>
        <button type="button" disabled={busy || blocked || dirty} onClick={() => void mutate("publish")}>Publish settings</button>
        <button type="button" disabled={busy || blocked} onClick={() => void mutate("discard")}>Restore published settings</button>
        <span>{busy ? "Saving…" : dirty ? "Unsaved changes" : "Draft saved"}</span>
      </div>
      <fieldset disabled={busy || blocked} style={{ border: 0, padding: 0, minWidth: 0 }}>
        <LeaderboardSettingsEditor clubId={clubId} document={doc} onChange={change} />
      </fieldset>
    </> : null}
  </section>;
}

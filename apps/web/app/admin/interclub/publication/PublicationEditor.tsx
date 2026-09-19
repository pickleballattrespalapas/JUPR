"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { apiError } from "@/lib/interclubRegistration";
import {
  type PublicMeet,
  type PublicLeague,
} from "@/lib/interclubPublic";
import PublicInterclubLeague from "@/components/PublicInterclubLeague";
import styles from "@/components/ClubWebsite.module.css";
type Publication = {
  revision: number;
  draft: { results: unknown[] };
  published: unknown;
};
type Context = {
  season: { details: { name: string; divisions: string[]; timezone: string } };
  clubs: { id: string; name: string }[];
  meets: PublicMeet[];
  publication: Publication;
  preview: PublicLeague;
  preview_fingerprint: string;
};
export default function PublicationEditor() {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const season = useSearchParams().get("season") || "";
  const allowed = session?.capabilities?.assignments.some(
    (a) =>
      a.club_id === clubId &&
      ["administrator", "super_admin", "club_owner"].includes(a.role),
  );
  if (loading) return <p>Checking access…</p>;
  if (!allowed || !accessToken)
    return <p>Organizer administrator access is required.</p>;
  if (!season)
    return (
      <p>
        <Link href="/admin/interclub">Choose a season first.</Link>
      </p>
    );
  return (
    <Editor
      key={`${clubId}:${season}:${session?.user?.id || session?.user?.email}`}
      clubId={clubId}
      season={season}
      accessToken={accessToken}
    />
  );
}
function Editor({
  clubId,
  season,
  accessToken,
}: {
  clubId: string;
  season: string;
  accessToken: string;
}) {
  const [data, setData] = useState<Context | null>(null),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [preview, setPreview] = useState(false),
    [reviewed, setReviewed] = useState(false),
    [revision, setRevision] = useState(0),
    [blocked, setBlocked] = useState(false);
  const token = useRef(accessToken);
  token.current = accessToken;
  const lock = useRef(false),
    pending = useRef<AbortController | null>(null);
  const endpoint = `${getAdminApiBaseUrl()}/admin/clubs/${encodeURIComponent(clubId)}/interclub/${encodeURIComponent(season)}/publication`;
  useEffect(() => {
    const controller = new AbortController();
    setData(null);
    setReviewed(false);
    setPreview(false);
    setError("");
    setBlocked(false);
    void fetch(endpoint, {
      headers: { Authorization: `Bearer ${token.current}` },
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (r) => {
        const d = await r.json();
        if (!r.ok)
          throw new Error(apiError(d, "Unable to load league publication."));
        if (!controller.signal.aborted) {
          setData(d);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [endpoint, revision]);
  useEffect(() => () => pending.current?.abort(), []);
  async function mutate(action: "save" | "publish" | "unpublish") {
    if (!data || lock.current || blocked) return;
    if (action === "publish" && !reviewed) return;
    if (readBrowserWorkspace()?.clubId !== clubId) {
      setError("The selected club changed. Reopen this workspace.");
      setBlocked(true);
      return;
    }
    lock.current = true;
    setBusy(true);
    setError("");
    const controller = new AbortController();
    pending.current = controller;
    let received = false;
    try {
      const r = await fetch(
        action === "save" ? endpoint : `${endpoint}/${action}`,
        {
          method: action === "save" ? "PUT" : "POST",
          headers: {
            Authorization: `Bearer ${token.current}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            revision: data.publication.revision,
            ...(action === "save" ? { document: data.publication.draft } : {}),
            ...(action === "publish" ? { preview_fingerprint: data.preview_fingerprint } : {}),
          }),
          signal: controller.signal,
        },
      );
      const d = await r.json();
      if (controller.signal.aborted) return;
      received = true;
      if (!r.ok) {
        if ([401, 403, 409].includes(r.status) || r.status >= 500)
          setBlocked(true);
        throw new Error(apiError(d, "Unable to save."));
      }
      setData({ ...data, publication: d });
      setMessage(
        action === "save"
          ? "League publication draft prepared. Preview before publishing."
          : action === "publish"
            ? "League website published."
            : "League website unpublished.",
      );
      setRevision((n) => n + 1);
    } catch (e) {
      if (!controller.signal.aborted) {
        if (!received) setBlocked(true);
        setError(e instanceof Error ? e.message : "Could not confirm save.");
      }
    } finally {
      if (!controller.signal.aborted) {
        lock.current = false;
        setBusy(false);
      }
    }
  }
  if (!data)
    return (
      <section>
        <h1>League website</h1>
        {error ? <p role="alert">{error}</p> : <p>Loading…</p>}
        <Link href="/admin/interclub">Back to seasons</Link>
      </section>
    );
  return (
    <section>
      <p>
        <Link href="/admin/interclub">← Interclub seasons</Link>
      </p>
      <h1>{data.season.details.name} · Public website</h1>
      <p>
        Preview and publish the whole league’s schedule, standings and results.
        Only organizer-approved meet results are included. Enter or correct scores
        in the meet workspace before publishing an updated snapshot.
      </p>
      <div className={styles.actions}>
        {!data.publication.revision && <button
          className={styles.primary}
          disabled={busy || blocked}
          onClick={() => void mutate("save")}
        >
          Prepare website preview
        </button>}
        <button
          className={styles.button}
          disabled={busy || blocked}
          onClick={() => { setPreview((v) => !v); setReviewed(true); }}
        >
          {preview ? "Close preview" : "Preview league website"}
        </button>
        <button
          className={styles.primary}
          disabled={busy || blocked || !reviewed || !data.publication.revision}
          onClick={() => void mutate("publish")}
        >
          Publish league
        </button>
        {!!data.publication.published && (
          <>
            <Link href={`/interclub/${season}`} target="_blank">
              View published league ↗
            </Link>
            <button
              className={styles.button}
              disabled={busy || blocked}
              onClick={() => void mutate("unpublish")}
            >
              Unpublish
            </button>
          </>
        )}
      </div>
      {!reviewed && <p>Open the league preview to review the current official results before publishing.</p>}
      {error && (
        <p role="alert" className={styles.error}>
          {error}
          {blocked && (
            <button
              className={styles.button}
              onClick={() => setRevision((n) => n + 1)}
            >
              Reload saved draft
            </button>
          )}
        </p>
      )}
      {message && (
        <p role="status" className={styles.notice}>
          {message}
        </p>
      )}
      {preview ? (
        <div className={styles.card} style={{ marginTop: 20 }}>
          <strong>Draft preview</strong>
          <PublicInterclubLeague league={data.preview} />
        </div>
      ) : (
        <section className={styles.card} style={{ marginTop: 20 }}>
          <h2>Official meet results</h2>
          <p>Use the meet workspace to print score sheets, enter both doubles pairings, and submit all official scores together. The organizer approves the exact submission before its results can appear here.</p>
          <Link className={styles.primary} href={`/admin/interclub/competition?season=${encodeURIComponent(season)}`}>Open meet scores & championships →</Link>
          <p>You can publish the schedule before any meet results exist. Publishing this website does not approve scores or update ratings.</p>
        </section>
      )}
    </section>
  );
}

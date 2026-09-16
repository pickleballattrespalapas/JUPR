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
  type Encounter,
  type PublicMeet,
  type PublicLeague,
  meetTime,
} from "@/lib/interclubPublic";
import PublicInterclubLeague from "@/components/PublicInterclubLeague";
import styles from "@/components/ClubWebsite.module.css";
type Publication = {
  revision: number;
  draft: { results: Encounter[] };
  published: unknown;
};
type Context = {
  season: { details: { name: string; divisions: string[]; timezone: string } };
  clubs: { id: string; name: string }[];
  meets: PublicMeet[];
  publication: Publication;
  preview: PublicLeague;
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
    [results, setResults] = useState<Encounter[]>([]),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [preview, setPreview] = useState(false),
    [revision, setRevision] = useState(0),
    [blocked, setBlocked] = useState(false);
  const token = useRef(accessToken);
  token.current = accessToken;
  const lock = useRef(false),
    pending = useRef<AbortController | null>(null);
  const endpoint = `${getAdminApiBaseUrl()}/admin/clubs/${encodeURIComponent(clubId)}/interclub/${encodeURIComponent(season)}/publication`;
  useEffect(() => {
    const controller = new AbortController();
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
          setResults(d.publication.draft.results);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, [endpoint, revision]);
  useEffect(() => () => pending.current?.abort(), []);
  const dirty =
    data &&
    JSON.stringify(results) !== JSON.stringify(data.publication.draft.results);
  async function mutate(action: "save" | "publish" | "unpublish") {
    if (!data || lock.current || blocked) return;
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
            ...(action === "save" ? { document: { results } } : {}),
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
          ? "Results draft saved. Preview before publishing."
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
  function edit(index: number, patch: Partial<Encounter>) {
    setResults((rows) =>
      rows.map((r, i) => (i === index ? { ...r, ...patch } : r)),
    );
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
        Publish the whole league’s schedule, standings and results. Clubs choose
        rosters per meet. Publishing results here does not apply rating updates.
      </p>
      <div className={styles.actions}>
        <button
          className={styles.primary}
          disabled={
            busy || blocked || (!dirty && data.publication.revision > 0)
          }
          onClick={() => void mutate("save")}
        >
          Save draft
        </button>
        <button
          className={styles.button}
          disabled={!!dirty}
          onClick={() => setPreview((v) => !v)}
        >
          {preview ? "Edit results" : "Preview saved draft"}
        </button>
        <button
          className={styles.primary}
          disabled={busy || blocked || !!dirty || !data.publication.revision}
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
      {dirty && <p>Unsaved changes. Save to update the preview.</p>}
      {preview ? (
        <div className={styles.card} style={{ marginTop: 20 }}>
          <strong>Draft preview</strong>
          <PublicInterclubLeague league={data.preview} />
        </div>
      ) : (
        <fieldset disabled={busy || blocked} style={{ border: 0, padding: 0 }}>
          <h2>Encounter results</h2>
          <p>
            Enter the three completed games for each club pairing and division.
            You can publish the schedule before any results exist.
          </p>
          {results.map((row, index) => {
            const meet = data.meets.find((m) => m.id === row.meet_id),
              clubs = data.clubs.filter((c) => meet?.club_ids.includes(c.id));
            return (
              <article
                className={`${styles.card} ${styles.form}`}
                style={{ margin: "1rem 0" }}
                key={row.id}
              >
                <h3>Encounter {index + 1}</h3>
                <div className={styles.grid}>
                  <label>
                    Meet
                    <select
                      value={row.meet_id}
                      onChange={(e) =>
                        edit(index, {
                          meet_id: e.target.value,
                          club_a: "",
                          club_b: "",
                        })
                      }
                    >
                      <option value="">Choose a meet</option>
                      {data.meets.map((m) => (
                        <option key={m.id} value={m.id}>
                          {meetTime(m.starts_at, data.season.details.timezone)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Division
                    <select
                      value={row.division}
                      onChange={(e) =>
                        edit(index, { division: e.target.value })
                      }
                    >
                      {data.season.details.divisions.map((d) => (
                        <option key={d}>{d}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className={styles.grid}>
                  {(["club_a", "club_b"] as const).map((side, i) => (
                    <label key={side}>
                      Club {i + 1}
                      <select
                        value={row[side]}
                        onChange={(e) =>
                          edit(index, { [side]: e.target.value })
                        }
                      >
                        <option value="">Choose club</option>
                        {clubs.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </div>
                <div className={styles.grid}>
                  {row.games.map((g, i) => (
                    <div key={i}>
                      <strong>Game {i + 1}</strong>
                      <label>
                        Club 1 score
                        <input
                          type="number"
                          min={0}
                          max={100}
                          value={g.a}
                          onChange={(e) =>
                            edit(index, {
                              games: row.games.map((old, j) =>
                                j === i
                                  ? { ...old, a: Number(e.target.value) }
                                  : old,
                              ),
                            })
                          }
                        />
                      </label>
                      <label>
                        Club 2 score
                        <input
                          type="number"
                          min={0}
                          max={100}
                          value={g.b}
                          onChange={(e) =>
                            edit(index, {
                              games: row.games.map((old, j) =>
                                j === i
                                  ? { ...old, b: Number(e.target.value) }
                                  : old,
                              ),
                            })
                          }
                        />
                      </label>
                    </div>
                  ))}
                </div>
                <button
                  className={styles.button}
                  onClick={() =>
                    setResults((rows) => rows.filter((_, i) => i !== index))
                  }
                >
                  Remove encounter from draft
                </button>
              </article>
            );
          })}
          <button
            className={styles.button}
            disabled={!data.meets.length}
            onClick={() =>
              setResults((rows) => [
                ...rows,
                {
                  id: crypto.randomUUID(),
                  meet_id: data.meets[0]?.id || "",
                  division: data.season.details.divisions[0],
                  club_a: "",
                  club_b: "",
                  games: [
                    { a: 0, b: 0 },
                    { a: 0, b: 0 },
                    { a: 0, b: 0 },
                  ],
                },
              ])
            }
          >
            + Add encounter result
          </button>
        </fieldset>
      )}
    </section>
  );
}

"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import {
  DISPLAY_LABELS,
  clubPageHref,
  type AdminSite,
  type DisplayKey,
  type SiteBlock,
  type SiteDocument,
  type SitePage,
} from "@/lib/clubSite";
import ClubSiteHeader from "@/components/ClubSiteHeader";
import ClubSiteContent from "@/components/ClubSiteContent";
import { ClubDisplayProvider, Display } from "@/components/ClubDisplay";
import styles from "@/components/ClubWebsite.module.css";
import editor from "./website.module.css";
import PageVisibilityEditor from "./PageVisibilityEditor";

export default function WebsitePage() {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const allowed = session?.capabilities?.assignments.some(
    (a) =>
      a.club_id === clubId &&
      ["administrator", "club_owner", "super_admin"].includes(a.role),
  );
  if (loading) return <p role="status">Checking club access…</p>;
  if (!accessToken || !allowed)
    return <p>Club administrator access is required to edit the website.</p>;
  return (
    <WebsiteEditor
      key={`${clubId}:${session?.user?.id || session?.user?.email || accessToken}`}
      clubId={clubId}
      accessToken={accessToken}
    />
  );
}

function WebsiteEditor({
  clubId,
  accessToken,
}: {
  clubId: string;
  accessToken: string;
}) {
  const [site, setSite] = useState<AdminSite | null>(null),
    [doc, setDoc] = useState<SiteDocument | null>(null);
  const [tab, setTab] = useState("identity"),
    [pageIndex, setPageIndex] = useState(0);
  const [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0),
    [mobile, setMobile] = useState(false);
  const token = useRef(accessToken);
  token.current = accessToken;
  const lock = useRef(false),
    pending = useRef<AbortController | null>(null);
  const api = getAdminApiBaseUrl(),
    endpoint = `${api}/admin/clubs/${encodeURIComponent(clubId)}/site`;
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
              : "Unable to load the club website.",
          );
        if (!controller.signal.aborted) {
          setSite(data);
          setDoc(data.draft);
          setPageIndex(0);
        }
      } catch (e) {
        if (!controller.signal.aborted)
          setError(
            e instanceof Error ? e.message : "Unable to load the club website.",
          );
      }
    })();
    return () => controller.abort();
  }, [endpoint, reload]);
  useEffect(() => () => pending.current?.abort(), []);
  const dirty =
    !!doc && !!site && JSON.stringify(doc) !== JSON.stringify(site.draft);
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
  async function mutate(action: "save" | "publish" | "unpublish" | "discard") {
    if (!site || !doc || lock.current || blocked) return;
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
            ...(action === "save" ? { document: doc } : {}),
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
              : "Check your website details.",
        );
      }
      setSite(data);
      setDoc(data.draft);
      setMessage(
        action === "save"
          ? "Draft saved. Your live website has not changed."
          : action === "publish"
            ? "Website published. Visitors can see this version now."
            : action === "unpublish"
              ? "Website unpublished. Your draft is still saved."
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
  function change(patch: Partial<SiteDocument>) {
    if (doc) {
      setDoc({ ...doc, ...patch });
      setMessage("");
    }
  }
  function updatePage(patch: Partial<SitePage>) {
    if (doc)
      change({
        pages: doc.pages.map((p, i) =>
          i === pageIndex ? { ...p, ...patch } : p,
        ),
      });
  }
  function updateBlock(index: number, patch: Partial<SiteBlock>) {
    if (doc)
      updatePage({
        blocks: doc.pages[pageIndex].blocks.map((b, i) =>
          i === index ? { ...b, ...patch } : b,
        ),
      });
  }
  function moveBlock(index: number, delta: number) {
    if (!doc) return;
    const blocks = [...doc.pages[pageIndex].blocks];
    [blocks[index], blocks[index + delta]] = [
      blocks[index + delta],
      blocks[index],
    ];
    updatePage({ blocks });
  }
  async function upload(file: File | undefined, apply: (url: string) => void) {
    if (!file) return;
    if (
      !["image/png", "image/jpeg", "image/webp"].includes(file.type) ||
      file.size > 220000
    ) {
      setError(
        "Choose a PNG, JPEG or WebP image under 220 KB. Use an HTTPS URL for larger images.",
      );
      return;
    }
    const reader = new FileReader();
    reader.onload = () => apply(String(reader.result));
    reader.readAsDataURL(file);
  }
  if (!site || !doc)
    return (
      <section>
        <h1>Club website</h1>
        {error ? (
          <p role="alert">
            {error}{" "}
            <button onClick={() => setReload((n) => n + 1)}>Reload</button>
          </p>
        ) : (
          <p role="status">Loading website…</p>
        )}
      </section>
    );
  const page = doc.pages[Math.min(pageIndex, doc.pages.length - 1)];
  return (
    <section>
      <p className={styles.eyebrow}>Your club’s public home</p>
      <h1>Club website</h1>
      <p>
        Edit your draft, preview it on desktop or mobile, then publish when
        you’re ready.
      </p>
      <div className={`${editor.toolbar} ${editor.stickyToolbar}`} >
        <div>
          <strong>{site.published ? "Published" : "Not published"}</strong>
          <br />
          <small>
            {dirty
              ? "Unsaved changes"
              : `Draft saved · version ${site.revision}`}
            {site.published &&
              ` · ${site.published.visibility === "listed" ? "Search engines allowed" : "Accessible by link"}`}
          </small>
        </div>
        <div className={styles.actions}>
          <button
            className={styles.button}
            disabled={busy || blocked || (!dirty && site.revision > 0)}
            onClick={() => void mutate("save")}
          >
            {busy ? "Please wait…" : "Save draft"}
          </button>
          <button className={styles.button} onClick={() => setTab("preview")}>
            Preview draft
          </button>
          <button
            className={styles.primary}
            disabled={
              busy || blocked || dirty || !site.revision || !site.club_active
            }
            onClick={() => void mutate("publish")}
          >
            Publish website
          </button>
        </div>
      </div>
      {!site.club_active && (
        <p className={styles.notice}>
          This club must be active before publishing.
        </p>
      )}
      {error && (
        <div role="alert" className={styles.error}>
          {error}
          {blocked && (
            <p>
              <button
                className={styles.button}
                onClick={() => setReload((n) => n + 1)}
              >
                Reload saved website
              </button>{" "}
              Reload replaces unsaved edits.
            </p>
          )}
        </div>
      )}
      {message && (
        <p role="status" className={styles.notice}>
          {message}
        </p>
      )}
      <nav className={editor.tabs} aria-label="Website editor">
        {[
          ["identity", "Club introduction"],
          ["pages", "Pages & layout"],
          ["visibility", "Page visibility"],
          ["display", "Stats & information"],
          ["leaderboard", "Overall leaderboard"],
          ["preview", "Preview & publish"],
        ].map(([value, label]) => (
          <button
            className={tab === value ? styles.primary : styles.button}
            key={value}
            aria-pressed={tab === value}
            onClick={() => setTab(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      <fieldset className={editor.fields} disabled={busy || blocked}>
        {tab === "identity" && (
          <div className={`${styles.card} ${styles.form}`}>
            <h2>Introduce your club</h2>
            <label>
              Public club name
              <input
                maxLength={120}
                value={doc.name}
                onChange={(e) => change({ name: e.target.value })}
              />
            </label>
            <label>
              Club description
              <textarea
                rows={5}
                maxLength={6000}
                value={doc.description}
                onChange={(e) => change({ description: e.target.value })}
              />
            </label>
            <label>
              Location
              <input
                maxLength={300}
                value={doc.location}
                onChange={(e) => change({ location: e.target.value })}
                placeholder="Town, region, country"
              />
            </label>
            <label>
              Visitor information
              <textarea
                rows={5}
                maxLength={6000}
                value={doc.visitor_info}
                onChange={(e) => change({ visitor_info: e.target.value })}
                placeholder="Where to play, opening times, drop-in arrangements and how to contact the club"
              />
            </label>
            <label>
              Club logo URL
              <input
                value={doc.logo_url.startsWith("data:") ? "" : doc.logo_url}
                onChange={(e) => change({ logo_url: e.target.value })}
                placeholder="https://…"
              />
            </label>
            <label>
              Or upload a logo
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(e) =>
                  void upload(e.target.files?.[0], (url) =>
                    change({ logo_url: url }),
                  )
                }
              />
            </label>
            {doc.logo_url && (
              <button
                className={styles.button}
                onClick={() => change({ logo_url: "" })}
              >
                Remove logo
              </button>
            )}
            <label>
              Accent color
              <input
                type="color"
                value={doc.accent}
                onChange={(e) => change({ accent: e.target.value })}
              />
            </label>
            <label>
              Website visibility
              <select
                value={doc.visibility}
                onChange={(e) =>
                  change({
                    visibility: e.target.value as SiteDocument["visibility"],
                  })
                }
              >
                <option value="listed">
                  Public — allow search engines to list this website
                </option>
                <option value="unlisted">
                  Link only — ask search engines not to list this website
                </option>
              </select>
            </label>
            <p>
              Anyone with the link can view an unlisted club. Visitors never
              need a player account. Visibility changes go live when you
              publish.
            </p>
          </div>
        )}
        {tab === "pages" && (
          <div className={editor.workspace}>
            <aside className={styles.card}>
              <h2>Pages</h2>
              <div className={styles.form}>
                {doc.pages.map((p, i) => (
                  <button
                    key={i}
                    className={pageIndex === i ? styles.primary : styles.button}
                    onClick={() => setPageIndex(i)}
                  >
                    {p.title || "Untitled page"}
                  </button>
                ))}
                <button
                  className={styles.button}
                  disabled={doc.pages.length >= 20}
                  onClick={() => {
                    let n = doc.pages.length;
                    while (doc.pages.some((p) => p.slug === `page-${n}`)) n++;
                    change({
                      pages: [
                        ...doc.pages,
                        {
                          slug: `page-${n}`,
                          title: "New page",
                          in_navigation: true,
                          blocks: [],
                        },
                      ],
                    });
                    setPageIndex(doc.pages.length);
                  }}
                >
                  + Add page
                </button>
              </div>
            </aside>
            <div className={`${styles.card} ${styles.form}`}>
              <h2>{page.title}</h2>
              <label>
                Page title
                <input
                  maxLength={80}
                  value={page.title}
                  onChange={(e) => updatePage({ title: e.target.value })}
                />
              </label>
              {page.slug !== "home" && (
                <>
                  <label>
                    Page address
                    <input
                      maxLength={60}
                      pattern="[a-z0-9]+(-[a-z0-9]+)*"
                      value={page.slug}
                      onChange={(e) =>
                        updatePage({ slug: e.target.value.toLowerCase() })
                      }
                    />
                  </label>
                  <label>
                    Page visibility
                    <select
                      value={page.in_navigation ? "public" : "private"}
                      onChange={(e) => updatePage({ in_navigation: e.target.value === "public" })}
                    >
                      <option value="public">Public</option>
                      <option value="private">Private · link only</option>
                    </select>
                  </label>
                  <div className={styles.actions}>
                    <button
                      className={styles.button}
                      onClick={() => {
                        change({
                          pages: doc.pages.filter((_, i) => i !== pageIndex),
                        });
                        setPageIndex(0);
                      }}
                    >
                      Remove page from draft
                    </button>
                    <button
                      className={styles.button}
                      disabled={pageIndex <= 1}
                      onClick={() => {
                        const pages = [...doc.pages];
                        [pages[pageIndex], pages[pageIndex - 1]] = [
                          pages[pageIndex - 1],
                          pages[pageIndex],
                        ];
                        change({ pages });
                        setPageIndex(pageIndex - 1);
                      }}
                    >
                      Move page earlier
                    </button>
                  </div>
                </>
              )}
              <p>
                Arrange blocks in rows of 12 columns. Blocks stack on phones.
                Your club introduction stays at the top of the home page.
              </p>
              {page.blocks.map((block, index) => (
                <article key={block.id} className={editor.blockEditor}>
                  <div className={styles.actions}>
                    <strong>
                      {index + 1}. {block.heading || block.kind}
                    </strong>
                    <button
                      className={styles.button}
                      disabled={!index}
                      aria-label={`Move block ${index + 1} up`}
                      onClick={() => moveBlock(index, -1)}
                    >
                      ↑
                    </button>
                    <button
                      className={styles.button}
                      disabled={index === page.blocks.length - 1}
                      aria-label={`Move block ${index + 1} down`}
                      onClick={() => moveBlock(index, 1)}
                    >
                      ↓
                    </button>
                    <button
                      className={styles.button}
                      onClick={() =>
                        updatePage({
                          blocks: page.blocks.filter((_, i) => i !== index),
                        })
                      }
                    >
                      Remove block
                    </button>
                  </div>
                  <label>
                    Heading
                    <input
                      value={block.heading}
                      maxLength={160}
                      onChange={(e) =>
                        updateBlock(index, { heading: e.target.value })
                      }
                    />
                  </label>
                  {["text", "button", "image"].includes(block.kind) && (
                    <label>
                      {block.kind === "button"
                        ? "Button text"
                        : block.kind === "image"
                          ? "Caption"
                          : "Text"}
                      <textarea
                        rows={block.kind === "text" ? 4 : 2}
                        maxLength={12000}
                        value={block.text}
                        onChange={(e) =>
                          updateBlock(index, { text: e.target.value })
                        }
                      />
                    </label>
                  )}
                  {["image", "button"].includes(block.kind) && (
                    <label>
                      {block.kind === "image"
                        ? "Image URL"
                        : "Link destination"}
                      <input
                        value={block.url.startsWith("data:") ? "" : block.url}
                        placeholder="https://…"
                        onChange={(e) =>
                          updateBlock(index, { url: e.target.value })
                        }
                      />
                    </label>
                  )}
                  {block.kind === "image" && (
                    <>
                      <label>
                        Or upload image
                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/webp"
                          onChange={(e) =>
                            void upload(e.target.files?.[0], (url) =>
                              updateBlock(index, { url }),
                            )
                          }
                        />
                      </label>
                      <label>
                        Image description
                        <input
                          maxLength={240}
                          value={block.alt}
                          onChange={(e) =>
                            updateBlock(index, { alt: e.target.value })
                          }
                        />
                      </label>
                    </>
                  )}
                  <div className={editor.row}>
                    <label>
                      Width
                      <select
                        value={block.span}
                        onChange={(e) =>
                          updateBlock(index, {
                            span: Number(e.target.value) as SiteBlock["span"],
                          })
                        }
                      >
                        <option value={12}>Full row</option>
                        <option value={8}>Two thirds</option>
                        <option value={6}>Half row</option>
                        <option value={4}>One third</option>
                        <option value={3}>One quarter</option>
                      </select>
                    </label>
                    <label>
                      Alignment
                      <select
                        value={block.align}
                        onChange={(e) =>
                          updateBlock(index, {
                            align: e.target.value as SiteBlock["align"],
                          })
                        }
                      >
                        {["left", "center", "right"].map((v) => (
                          <option key={v}>{v}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Background
                      <select
                        value={block.tone}
                        onChange={(e) =>
                          updateBlock(index, {
                            tone: e.target.value as SiteBlock["tone"],
                          })
                        }
                      >
                        {["plain", "soft", "accent"].map((v) => (
                          <option key={v}>{v}</option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Spacing
                      <select
                        value={block.padding}
                        onChange={(e) =>
                          updateBlock(index, {
                            padding: e.target.value as SiteBlock["padding"],
                          })
                        }
                      >
                        {["small", "medium", "large"].map((v) => (
                          <option key={v}>{v}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                </article>
              ))}
              <div className={styles.actions}>
                {(
                  [
                    "text",
                    "image",
                    "button",
                    "links",
                    "divider",
                  ] as SiteBlock["kind"][]
                ).map((kind) => (
                  <button
                    className={styles.button}
                    key={kind}
                    disabled={page.blocks.length >= 40}
                    onClick={() =>
                      updatePage({
                        blocks: [
                          ...page.blocks,
                          {
                            id: crypto.randomUUID(),
                            kind,
                            heading: "",
                            text: "",
                            url: "",
                            alt: "",
                            span: 12,
                            align: "left",
                            tone: "plain",
                            padding: "medium",
                          },
                        ],
                      })
                    }
                  >
                    + {kind === "links" ? "Club links" : kind}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {tab === "visibility" && (
          <PageVisibilityEditor site={site} document={doc} onChange={change} />
        )}
        {tab === "leaderboard" && (
          <section className={styles.card}>
            <h2>Overall leaderboard</h2>
            <p>Cards, statistics and seasons have their own settings. Save and publish them independently of your website draft.</p>
            <Link className={styles.primary} href="/admin/leaderboard-settings">Edit leaderboard settings</Link>
          </section>
        )}
        {tab === "display" && (
          <div className={`${styles.card} ${styles.form}`}>
            <h2>Choose what visitors see</h2>
            <p>
              These choices apply to standard player, leaderboard, registration
              and results pages. They are display settings; club records stay
              available to staff.
            </p>
            <p>
              Featured cards on the Overall leaderboard are selected separately
              under Overall leaderboard. Hiding a statistic here does not hide
              a card you selected there.
            </p>
            <div className={styles.grid}>
              {(Object.entries(DISPLAY_LABELS) as [DisplayKey, string][]).map(
                ([key, label]) => (
                  <label key={key}>
                    <span>
                      <input
                        type="checkbox"
                        checked={doc.display[key] !== false}
                        onChange={(e) =>
                          change({
                            display: {
                              ...doc.display,
                              [key]: e.target.checked,
                            },
                          })
                        }
                      />{" "}
                      {label}
                    </span>
                  </label>
                ),
              )}
            </div>
          </div>
        )}
        {tab === "preview" && (
          <>
            <div className={styles.notice}>
              <strong>Draft preview — only you can see these edits</strong>
              <p>
                {dirty
                  ? "Save your draft before publishing."
                  : "Your saved draft is ready to publish."}{" "}
                {doc.visibility === "listed"
                  ? "Search engines may list the public pages of this website."
                  : "Visitors can open the website by link. Search engines will be asked not to list it."}
              </p>
              <div className={styles.actions}>
                <button type="button" className={styles.button} onClick={() => setTab("visibility")}>
                  Edit page visibility
                </button>
                <label>
                  Preview page{" "}
                  <select
                    value={pageIndex}
                    onChange={(e) => setPageIndex(Number(e.target.value))}
                  >
                    {doc.pages.map((p, i) => (
                      <option value={i} key={i}>
                        {p.title}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className={styles.button}
                  aria-pressed={mobile}
                  onClick={() => setMobile((v) => !v)}
                >
                  {mobile ? "Desktop preview" : "Mobile preview"}
                </button>
              </div>
            </div>
            <div
              className={editor.preview}
              style={{ maxWidth: mobile ? 390 : "100%" }}
            >
              <ClubSiteHeader document={doc} slug={site.slug} />
              <ClubSiteContent
                document={doc}
                slug={site.slug}
                pageSlug={page.slug}
              />
            </div>
            <section className={styles.card}>
              <h2>Statistics preview</h2>
              <p>
                Illustrative display only — these are sample figures, not club
                records.
              </p>
              <ClubDisplayProvider display={doc.display}>
                <div className={styles.actions}>
                  <strong>Sample player</strong>
                  <Display field="ratings">
                    <span>Rating: 3.50</span>
                  </Display>
                  <Display field="records">
                    <span>Record: 8–4</span>
                  </Display>
                  <Display field="match_counts">
                    <span>Matches: 12</span>
                  </Display>
                  <Display field="win_percentage">
                    <span>Win rate: 66.7%</span>
                  </Display>
                  <Display field="rating_changes">
                    <span>Gain: +0.12</span>
                  </Display>
                </div>
              </ClubDisplayProvider>
            </section>
          </>
        )}
      </fieldset>
      <div className={editor.toolbar}>
        <div>
          {site.published && (
            <Link
              href={clubPageHref(site.slug, "home")}
              target="_blank"
              rel="noreferrer"
            >
              View published website ↗
            </Link>
          )}
        </div>
        <div className={styles.actions}>
          {site.published && (
            <>
              <button
                className={styles.button}
                disabled={busy || blocked}
                onClick={() => void mutate("discard")}
              >
                Restore published version
              </button>
              <button
                className={styles.button}
                disabled={busy || blocked || dirty}
                onClick={() => void mutate("unpublish")}
              >
                Unpublish website
              </button>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

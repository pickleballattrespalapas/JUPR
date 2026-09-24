"use client";
import { useEffect, useState } from "react";
import {
  CLUB_LINKS, PAGE_SCOPE_LABELS, clubPageHref,
  type AdminSite, type PageVisibility, type SiteDocument,
} from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";

export default function PageVisibilityEditor({ site, document: doc, onChange }: {
  site: AdminSite; document: SiteDocument; onChange: (patch: Partial<SiteDocument>) => void;
}) {
  const [origin, setOrigin] = useState("");
  const [notice, setNotice] = useState("");
  useEffect(() => setOrigin(window.location.origin), []);
  const rows = [
    ...CLUB_LINKS.map(([label, key]) => ({
      key, label, description: PAGE_SCOPE_LABELS[key],
      href: `/clubs/${encodeURIComponent(site.slug)}/${key}`,
      value: doc.page_visibility?.[key] || "public",
      live: site.published ? site.published.page_visibility?.[key] || "public" : null,
      change: (value: PageVisibility) => onChange({ page_visibility: { ...doc.page_visibility, [key]: value } }),
    })),
    ...doc.pages.filter((p) => p.slug !== "home").map((p) => {
      const live = site.published?.pages.find((old) => old.slug === p.slug);
      return {
        key: `pages/${p.slug}`, label: p.title, description: "Custom page",
        href: clubPageHref(site.slug, p.slug), value: p.in_navigation ? "public" : "private",
        live: live ? (live.in_navigation ? "public" : "private") : null,
        change: (value: PageVisibility) => onChange({ pages: doc.pages.map((old) => old.slug === p.slug ? { ...old, in_navigation: value === "public" } : old) }),
      };
    }),
  ];
  async function copy(href: string, label: string) {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}${href}`);
      setNotice(`Link copied for ${label}.`);
    } catch {
      setNotice(`Select and copy the link shown for ${label}.`);
    }
  }
  return <section>
    <h2>Choose which pages are public</h2>
    <p><strong>Public</strong> pages appear in your club’s navigation and homepage links. <strong>Private · link only</strong> pages open for anyone with the link, without a login, and are hidden from navigation and search indexing.</p>
    <p>Save and publish to apply these choices. Shared links open the current published version. A section’s setting also applies to its related pages.</p>
    {notice && <p role="status" className={styles.notice}>{notice}</p>}
    <div className={styles.grid}>
      {rows.map((row) => <section key={row.key} className={`${styles.card} ${styles.form}`}>
        <div><h3 style={{ margin: 0 }}>{row.label}</h3><p style={{ margin: ".5rem 0" }}>{row.description}</p>
          <small>{row.live ? `Live: ${row.live === "public" ? "Public" : "Private · link only"}` : "Not published"}</small>
        </div>
        <label>Visibility
          <select aria-label={`Visibility for ${row.label}`} value={row.value} onChange={(e) => row.change(e.target.value as PageVisibility)}>
            <option value="public">Public</option><option value="private">Private · link only</option>
          </select>
        </label>
        <label>Share link
          <input aria-label={`Link to ${row.label}`} readOnly value={`${origin}${row.href}`} onFocus={(e) => e.target.select()} />
        </label>
        <button type="button" className={styles.button} disabled={!row.live || !site.club_active} aria-label={`Copy link to ${row.label}`} onClick={() => void copy(row.href, row.label)}>Copy link</button>
        {!row.live && <small>Publish the website and this page before sharing.</small>}
      </section>)}
    </div>
  </section>;
}

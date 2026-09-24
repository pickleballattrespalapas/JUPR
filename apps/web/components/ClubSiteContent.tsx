/* Images may be uploaded or hosted by clubs; render only validated sources. */
/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import {
  publicClubLinks,
  canLinkClubPage,
  clubPageHref,
  accentTextColor,
  safeImageUrl,
  safeSiteUrl,
  type SiteDocument,
} from "@/lib/clubSite";
import styles from "./ClubWebsite.module.css";
export default function ClubSiteContent({
  document: doc,
  slug,
  pageSlug = "home",
}: {
  document: SiteDocument;
  slug: string;
  pageSlug?: string;
}) {
  const page = doc.pages.find((p) => p.slug === pageSlug);
  const links = publicClubLinks(doc);
  if (!page) return <p>This page is unavailable.</p>;
  return (
    <div>
      {pageSlug === "home" ? (
        <section
          className={styles.hero}
          style={{ borderTop: `5px solid ${doc.accent}` }}
        >
          <p className={styles.eyebrow}>
            {doc.location || "Welcome to our club"}
          </p>
          <h1>{doc.name}</h1>
          {doc.description && <p>{doc.description}</p>}
          {doc.visitor_info && (
            <>
              <h2>Planning a visit?</h2>
              <p>{doc.visitor_info}</p>
            </>
          )}
        </section>
      ) : (
        <h1>{page.title}</h1>
      )}
      <div className={styles.blocks}>
        {page.blocks.map((block) => (
          <section
            key={block.id}
            className={styles.block}
            style={{
              gridColumn: `span ${block.span}`,
              textAlign: block.align,
              padding: { small: ".5rem", medium: "1.5rem", large: "3rem" }[
                block.padding
              ],
              background:
                block.tone === "accent"
                  ? doc.accent
                  : block.tone === "soft"
                    ? "#eff6ff"
                    : "white",
              color: block.tone === "accent" ? accentTextColor(doc.accent) : "#0f172a",
            }}
          >
            {block.heading && <h2>{block.heading}</h2>}
            {block.kind === "text" && (
              <p style={{ marginBottom: 0 }}>{block.text}</p>
            )}
            {block.kind === "image" && safeImageUrl(block.url) && (
              <figure style={{ margin: 0 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={safeImageUrl(block.url)}
                  alt={block.alt}
                  loading="lazy"
                />
                {block.text && <figcaption>{block.text}</figcaption>}
              </figure>
            )}
            {block.kind === "button" && safeSiteUrl(block.url) && canLinkClubPage(doc, slug, block.url, clubPageHref(slug, pageSlug)) && (
              <a className={styles.button} href={safeSiteUrl(block.url)}>
                {block.text || "Learn more"}
              </a>
            )}
            {block.kind === "divider" && <hr />}
            {block.kind === "links" && (
              <div className={styles.actions}>
                {links.map(([label, path]) => (
                  <Link
                    className={styles.button}
                    key={path}
                    href={`/clubs/${slug}/${path}`}
                  >
                    {label}
                  </Link>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
      {pageSlug === "home" && links.length > 0 && (
        <section>
          <h2>Around the club</h2>
          <div className={styles.grid}>
            {links.map(([label, path]) => (
              <Link
                key={path}
                className={styles.cardLink}
                href={`/clubs/${slug}/${path}`}
              >
                <strong>{label}</strong>
                <small>Explore →</small>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

import Link from "next/link";
import {
  publicClubLinks,
  clubPageHref,
  safeImageUrl,
  type SiteDocument,
} from "@/lib/clubSite";
import styles from "./ClubWebsite.module.css";
export default function ClubSiteHeader({
  document: doc,
  slug,
}: {
  document: SiteDocument;
  slug: string;
}) {
  return (
    <>
      <header className={styles.header}>
        <Link href={`/clubs/${slug}`} className={styles.brand}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          {safeImageUrl(doc.logo_url) && (
            <img
              className={styles.logo}
              src={safeImageUrl(doc.logo_url)}
              alt=""
            />
          )}
          {doc.name}
        </Link>
        <div className={styles.actions}>
          <Link className={styles.button} href="/clubs">
            Change club
          </Link>
          <Link href="/admin/login">Staff sign in</Link>
        </div>
      </header>
      <nav className={styles.nav} aria-label="Club navigation">
        <Link href={`/clubs/${slug}`}>Home</Link>
        {publicClubLinks(doc).map(([label, path]) => (
          <Link key={path} href={`/clubs/${slug}/${path}`}>
            {label}
          </Link>
        ))}
        {doc.pages
          .filter((p) => p.slug !== "home" && p.in_navigation)
          .map((p) => (
            <Link key={p.slug} href={clubPageHref(slug, p.slug)}>
              {p.title}
            </Link>
          ))}
      </nav>
    </>
  );
}

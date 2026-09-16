import Link from "next/link";
import { getClubDirectory, type Directory } from "@/lib/clubSiteServer";
import { safeImageUrl } from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";
export const dynamic = "force-dynamic";
export default async function DirectoryPage({
  searchParams,
}: {
  searchParams?: { q?: string; page?: string };
}) {
  const q = String(searchParams?.q || "").slice(0, 120);
  const page = Math.max(1, Math.min(10000, Number(searchParams?.page) || 1));
  let data: Directory | null = null,
    error = false;
  try {
    data = await getClubDirectory(q, (page - 1) * 100);
  } catch {
    error = true;
  }
  return (
    <section className={styles.page}>
      <p className={styles.eyebrow}>Find your community</p>
      <h1>Find my club</h1>
      <p>Search by name or browse the alphabetical directory.</p>
      <form className={styles.form} action="/clubs" style={{ maxWidth: 640 }}>
        <label>
          Club name
          <input
            type="search"
            name="q"
            defaultValue={q}
            placeholder="Search clubs"
            maxLength={120}
          />
        </label>
        <div className={styles.actions}>
          <button className={styles.primary}>Search</button>
          {q && <Link href="/clubs">Clear search</Link>}
        </div>
      </form>
      {error && (
        <p role="alert" className={styles.error}>
          The directory could not load. <Link href="/clubs">Try again</Link>.
        </p>
      )}
      {data && (
        <>
          <p>
            {data.total} {data.total === 1 ? "club" : "clubs"}
            {q ? ` matching “${q}”` : " · A–Z"}
          </p>
          <div className={styles.grid}>
            {data.clubs.map((club) => (
              <Link
                className={styles.cardLink}
                key={club.slug}
                href={`/clubs/${club.slug}`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {safeImageUrl(club.logo_url) && (
                  <img
                    className={styles.logo}
                    src={safeImageUrl(club.logo_url)}
                    alt=""
                  />
                )}
                <h2>{club.name}</h2>
                {club.location && <p>{club.location}</p>}
                <p>{club.description.slice(0, 180)}</p>
                <small>Visit club →</small>
              </Link>
            ))}
          </div>
          {!data.clubs.length && (
            <p>
              {q
                ? "No clubs match that name."
                : "Clubs will appear here when their administrators publish a listed website."}{" "}
              If your club uses an unlisted website, ask its administrator for
              the link.
            </p>
          )}
          <nav className={styles.actions} aria-label="Directory pages">
            {page > 1 && (
              <Link
                className={styles.button}
                href={`/clubs?q=${encodeURIComponent(q)}&page=${page - 1}`}
              >
                Previous
              </Link>
            )}
            {page * 100 < data.total && (
              <Link
                className={styles.button}
                href={`/clubs?q=${encodeURIComponent(q)}&page=${page + 1}`}
              >
                Next
              </Link>
            )}
          </nav>
        </>
      )}
      <section className={styles.card} style={{ marginTop: "2rem" }}>
        <h2>Bring your club to PCS</h2>
        <p>
          Create your club, add its players and programs, then publish when
          you’re ready.
        </p>
        <Link className={styles.primary} href="/create-club">
          Create a club
        </Link>
      </section>
    </section>
  );
}

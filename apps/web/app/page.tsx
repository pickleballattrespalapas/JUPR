import Link from "next/link";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { getPublicSite } from "@/lib/clubSiteServer";
import { LAST_CLUB_COOKIE, validClubSlug } from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";
export const dynamic = "force-dynamic";
export default async function Home({ searchParams }: { searchParams?: { welcome?: string } }) {
  const slug = cookies().get(LAST_CLUB_COOKIE)?.value || "";
  let remembered = false;
  if (!searchParams?.welcome && validClubSlug(slug)) {
    try { remembered = Boolean(await getPublicSite(slug)); } catch { /* Homepage remains available during an API outage. */ }
  }
  if (remembered) redirect(`/clubs/${slug}`);
  return <section className={styles.page}>
    <div className={styles.hero}><p className={styles.eyebrow}>Pickleball Club Sandwich</p><h1>Your club. Your players.<br/>More time on court.</h1>
      <p>A home for your pickleball community. Find club information, follow player ratings and match results, and keep up with leagues and tournaments.</p>
      <div className={styles.actions}><Link className={styles.primary} href="/clubs">Find my club →</Link><Link className={styles.button} href="/create-club">Create a club</Link></div>
    </div>
    <div className={styles.grid}>
      <article className={styles.card}><h2>A home for every club</h2><p>Discover your club’s people, programs and visitor information. Everyone can browse; players don’t need an account.</p></article>
      <article className={styles.card}><h2>Follow the action</h2><p>Explore ratings, player profiles, match history and competitions, all in your club’s own space.</p></article>
      <article className={styles.card}><h2>Connect through competition</h2><p>Follow interclub leagues from participating clubs, with one view of the whole league’s schedule, results and standings.</p></article>
    </div>
    <section className={styles.card}><h2>Run a club?</h2><p>Set up your club website, manage players and events, and choose when to publish. Customize your pages and the information your community sees.</p><div className={styles.actions}><Link className={styles.primary} href="/create-club">Get started</Link><Link href="/admin/login">Staff sign in</Link></div></section>
  </section>;
}

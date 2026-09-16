import Link from "next/link";
import styles from "@/components/ClubWebsite.module.css";
export default function SiteMapPage(){return <section className={styles.page}><h1>Site map</h1><div className={styles.grid}>
  <section className={styles.card}><h2>Clubs</h2><p><Link href="/clubs">Search clubs and browse the alphabetical directory</Link></p><p><Link href="/create-club">Create a club</Link></p><p>Open a club to find its player profiles, leaderboards, matches, events, custom pages and interclub leagues.</p></section>
  <section className={styles.card}><h2>About PCS</h2>{[["About Pickleball Club Sandwich","/?welcome=1"],["How ratings work","/how-ratings-work"],["FAQ","/faq"],["Support","/support"],["Privacy","/privacy"],["Terms","/terms"],["Data corrections","/data-corrections"]].map(([label,href])=><p key={href}><Link href={href}>{label}</Link></p>)}</section>
  <section className={styles.card}><h2>Club staff</h2><p><Link href="/admin/login">Staff sign in</Link></p><p><Link href="/admin/select-club">Choose club workspace</Link></p><p><Link href="/admin/website">Edit club website</Link></p></section>
  </div></section>;}

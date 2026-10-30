import { requestShareMetadata } from "@/lib/shareMetadata";
import { cache, Suspense, type CSSProperties, type ReactNode } from "react";
import Link from "next/link";
import { headers } from "next/headers";
import { getPublicSiteShell } from "@/lib/clubSiteServer";
import { publicClubPage, type PublicSiteShell } from "@/lib/clubSite";
import PublicSiteHeader from "@/components/PublicSiteHeader";
import { InteractionProvider } from "@/components/interaction";

const productName = "Pickleball Club Sandwich";

function deploymentGitSha(): string | null {
  const value = String(process.env.VERCEL_GIT_COMMIT_SHA || "")
    .trim()
    .toLowerCase();
  return /^[0-9a-f]{40}$/.test(value) ? value : null;
}

export async function generateMetadata() {
  return requestShareMetadata("/", true);
}

const shellStyle: CSSProperties = {
  margin: "0 auto",
  maxWidth: "1380px",
  padding: "1rem",
  fontFamily:
    "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  lineHeight: 1.5
};

const footerStyle: CSSProperties = {
  marginTop: "2.5rem",
  paddingTop: "1rem",
  borderTop: "1px solid #e2e8f0",
  display: "flex",
  justifyContent: "space-between",
  gap: "0.75rem",
  flexWrap: "wrap",
  fontSize: "0.9rem"
};

// Global navigation has no club dependency on club/admin routes. On global
// pages, its small published-settings read streams independently of page content.
const globalNavigationSite = cache(async () => {
  const path = (await headers()).get("x-pcs-club-path") || "/";
  if (path.startsWith("/admin") || path.startsWith("/clubs/")) return null;
  return getPublicSiteShell("tres-palapas").catch(() => null);
});

async function HeaderNavigation(props: { productName: string; isStaging: boolean; stagingBuildSha: string | null }) {
  const site = await globalNavigationSite();
  return <PublicSiteHeader {...props} clubDocument={site?.document || null} />;
}

function FooterLinks({ site }: { site?: PublicSiteShell | null }) {
  const showClubPage = (path: string) => !!site && !!publicClubPage(site.document, site.slug, path);
  return (
    <nav style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }} aria-label="Footer navigation">
      {showClubPage("/clubs/tres-palapas/leagues") && <Link href="/clubs/tres-palapas/leagues">Leagues</Link>}
      {showClubPage("/clubs/tres-palapas/tournaments") && <Link href="/clubs/tres-palapas/tournaments">Tournaments</Link>}
      <Link href="/admin/login">Staff sign in</Link>
      <Link href="/site-map">Site map</Link>
      {showClubPage("/clubs/tres-palapas/badge-codex") && <Link href="/clubs/tres-palapas/badge-codex">Badges & Trophies</Link>}
      {showClubPage("/clubs/tres-palapas/matches") && <Link href="/clubs/tres-palapas/matches">Matches</Link>}
      <Link href="/how-ratings-work">How ratings work</Link>
      <Link href="/faq">FAQ</Link>
      <Link href="/privacy">Privacy</Link>
      <Link href="/terms">Terms</Link>
      <Link href="/support">Contact</Link>
      <Link href="/data-corrections">Data corrections</Link>
    </nav>
  );
}

async function FooterNavigation() {
  return <FooterLinks site={await globalNavigationSite()} />;
}

export default function RootLayout({ children }: { children: ReactNode }) {
  const isStaging =
    (process.env.NEXT_PUBLIC_JUPR_ENV || "").trim().toLowerCase() ===
    "staging";
  const stagingBuildSha = isStaging ? deploymentGitSha() : null;

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#f8fafc", color: "#0f172a" }}>
        <InteractionProvider>
          <div style={shellStyle}>
            <Suspense fallback={<PublicSiteHeader productName={productName} isStaging={isStaging} stagingBuildSha={stagingBuildSha} />}>
              <HeaderNavigation productName={productName} isStaging={isStaging} stagingBuildSha={stagingBuildSha} />
            </Suspense>
            <main style={{ minWidth: 0 }}>{children}</main>
            <footer style={footerStyle}>
              <span style={{ color: "#475569" }}>
                Follow your club’s ratings, matches, leagues, and events.
              </span>
              <Suspense fallback={<FooterLinks />}>
                <FooterNavigation />
              </Suspense>
            </footer>
          </div>
        </InteractionProvider>
      </body>
    </html>
  );
}

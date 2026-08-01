import type { Metadata } from "next";
import type { CSSProperties, ReactNode } from "react";
import Link from "next/link";
import PublicSiteHeader from "@/components/PublicSiteHeader";

const productName = "Pickleball Club Sandwich";

export const metadata: Metadata = {
  title: productName,
  description:
    "Club websites, live scoring, ratings, leaderboards, player profiles, and event scoring for pickleball clubs."
};

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

export default function RootLayout({ children }: { children: ReactNode }) {
  const isStaging =
    (process.env.NEXT_PUBLIC_JUPR_ENV || "").trim().toLowerCase() ===
    "staging";

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#f8fafc", color: "#0f172a" }}>
        <div style={shellStyle}>
          <PublicSiteHeader productName={productName} isStaging={isStaging} />
          <main style={{ minWidth: 0 }}>{children}</main>
          <footer style={footerStyle}>
            <span style={{ color: "#475569" }}>
              {productName} is the live ratings and event layer for pickleball
              clubs.
            </span>
            <nav
              style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}
              aria-label="Footer navigation"
            >
              <Link href="/clubs/tres-palapas/leagues">Leagues</Link>
              <Link href="/clubs/tres-palapas/tournaments">Tournaments</Link>
              <Link href="/admin/login">Staff sign in</Link>
              <Link href="/site-map">Site Map</Link>
              <Link href="/clubs/tres-palapas/badge-codex">Badge Codex</Link>
              <Link href="/clubs/tres-palapas/matches">Matches</Link>
              <Link href="/how-ratings-work">How ratings work</Link>
              <Link href="/faq">FAQ</Link>
              <Link href="/privacy">Privacy</Link>
              <Link href="/terms">Terms</Link>
              <Link href="/support">Contact</Link>
              <Link href="/data-corrections">Data corrections</Link>
            </nav>
          </footer>
        </div>
      </body>
    </html>
  );
}

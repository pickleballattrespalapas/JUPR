import type { Metadata } from "next";
import type { CSSProperties } from "react";
import Link from "next/link";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "JUPR Leagues",
  description: "Live pickleball ratings, leaderboards, player profiles, and event scoring for clubs."
};

const shellStyle: CSSProperties = {
  margin: "0 auto",
  maxWidth: "1120px",
  padding: "1rem",
  fontFamily: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  lineHeight: 1.5
};

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "1.5rem",
  gap: "0.75rem",
  flexWrap: "wrap"
};

const navStyle: CSSProperties = {
  display: "flex",
  gap: "1rem",
  fontSize: "0.95rem",
  flexWrap: "wrap",
  alignItems: "center"
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
  const isStaging = (process.env.NEXT_PUBLIC_JUPR_ENV || "").trim().toLowerCase() === "staging";

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#f8fafc", color: "#0f172a" }}>
        <div style={shellStyle}>
          <header style={headerStyle}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Link href="/" style={{ fontWeight: 800, color: "inherit", textDecoration: "none", letterSpacing: "-0.02em" }}>
                JUPR Leagues
              </Link>
              {isStaging ? (
                <span style={{ fontSize: "0.75rem", padding: "0.15rem 0.4rem", background: "#fef3c7", borderRadius: "4px" }}>
                  STAGING
                </span>
              ) : null}
            </div>
            <nav style={navStyle} aria-label="Primary navigation">
              <Link href="/">Home</Link>
              <Link href="/clubs/tres-palapas">Club</Link>
              <Link href="/clubs/tres-palapas/live">Live</Link>
              <Link href="/clubs/tres-palapas/leaderboards">Leaderboards</Link>
              <Link href="/clubs/tres-palapas/match-explorer">Match Explorer</Link>
              <Link href="/clubs/tres-palapas/weekly-recap">Weekly Recap</Link>
              <Link href="/clubs/tres-palapas/tournament-registration">Register</Link>
              <Link href="/clubs/tres-palapas/tournament-roster">Roster</Link>
              <Link href="/clubs/tres-palapas/tournament-partner-board">Partner Board</Link>
              <Link href="/clubs/tres-palapas/players">Players</Link>
              <Link href="/admin">Admin</Link>
            </nav>
          </header>
          <main>{children}</main>
          <footer style={footerStyle}>
            <span style={{ color: "#475569" }}>JUPR is the live ratings and event layer for pickleball clubs.</span>
            <nav style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }} aria-label="Footer navigation">
              <Link href="/clubs/tres-palapas/league-results">League Results</Link>
              <Link href="/clubs/tres-palapas/badge-codex">Badge Codex</Link>
              <Link href="/clubs/tres-palapas/challenge-ladder">Challenge Ladder</Link>
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

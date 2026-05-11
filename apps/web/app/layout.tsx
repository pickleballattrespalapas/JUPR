import type { Metadata } from "next";
import type { CSSProperties } from "react";
import Link from "next/link";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "JUPR Public Web",
  description: "Public, read-only JUPR pages for clubs and leaderboards"
};

const shellStyle: CSSProperties = {
  margin: "0 auto",
  maxWidth: "960px",
  padding: "1rem",
  fontFamily: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  lineHeight: 1.5
};

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "1rem",
  gap: "0.75rem",
  flexWrap: "wrap"
};

const footerStyle: CSSProperties = {
  marginTop: "2rem",
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
              <Link href="/" style={{ fontWeight: 700, color: "inherit", textDecoration: "none" }}>
                JUPR
              </Link>
              {isStaging ? (
                <span style={{ fontSize: "0.75rem", padding: "0.15rem 0.4rem", background: "#fef3c7", borderRadius: "4px" }}>
                  STAGING
                </span>
              ) : null}
            </div>
            <nav style={{ display: "flex", gap: "1rem", fontSize: "0.95rem" }}>
              <Link href="/">Home</Link>
              <Link href="/clubs/tres-palapas">Tres Palapas</Link>
              <Link href="/clubs/tres-palapas/leaderboards">Leaderboards</Link>
            </nav>
          </header>
          <main>{children}</main>
          <footer style={footerStyle}>
            <span style={{ color: "#475569" }}>Read-only public preview</span>
            <nav style={{ display: "flex", gap: "1rem" }}>
              <Link href="https://trespalapasresort.com/privacy-policy" target="_blank" rel="noreferrer">Privacy</Link>
              <Link href="https://trespalapasresort.com/terms-conditions" target="_blank" rel="noreferrer">Terms</Link>
              <Link href="mailto:hello@jupr.app">Contact</Link>
            </nav>
          </footer>
        </div>
      </body>
    </html>
  );
}

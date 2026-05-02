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
  maxWidth: "860px",
  padding: "1rem",
  fontFamily: "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  lineHeight: 1.5
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#f8fafc", color: "#0f172a" }}>
        <div style={shellStyle}>
          <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <Link href="/" style={{ fontWeight: 700, color: "inherit", textDecoration: "none" }}>
              JUPR
            </Link>
            <nav style={{ display: "flex", gap: "1rem", fontSize: "0.95rem" }}>
              <Link href="/clubs/tres-palapas">Club</Link>
              <Link href="/clubs/tres-palapas/leaderboards">Leaderboards</Link>
            </nav>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}

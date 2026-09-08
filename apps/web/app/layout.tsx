import type { Metadata } from "next";
import type { CSSProperties, ReactNode } from "react";
import PublicFooterNav from "@/components/PublicFooterNav";
import PublicSiteHeader from "@/components/PublicSiteHeader";
import { InteractionProvider } from "@/components/interaction";

const productName = "Pickleball Club Sandwich";

function deploymentGitSha(): string | null {
  const value = String(process.env.VERCEL_GIT_COMMIT_SHA || "")
    .trim()
    .toLowerCase();
  return /^[0-9a-f]{40}$/.test(value) ? value : null;
}

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
  const stagingBuildSha = isStaging ? deploymentGitSha() : null;

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#f8fafc", color: "#0f172a" }}>
        <InteractionProvider>
          <div style={shellStyle}>
            <PublicSiteHeader
              productName={productName}
              isStaging={isStaging}
              stagingBuildSha={stagingBuildSha}
            />
            <main style={{ minWidth: 0 }}>{children}</main>
            <footer style={footerStyle}>
              <span style={{ color: "#475569" }}>
                Follow your club’s ratings, matches, leagues, and events.
              </span>
              <PublicFooterNav />
            </footer>
          </div>
        </InteractionProvider>
      </body>
    </html>
  );
}

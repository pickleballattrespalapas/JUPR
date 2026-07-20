import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function safeOrigin(value: string | undefined): string | null {
  const raw = String(value || "").trim();
  if (!raw) return null;
  try {
    const parsed = new URL(raw);
    if (!["http:", "https:"].includes(parsed.protocol)) return null;
    return parsed.origin;
  } catch {
    return null;
  }
}

function truthy(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes(String(value || "").trim().toLowerCase());
}

export async function GET() {
  const apiOrigin = safeOrigin(
    process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL
  );
  const vercelEnvironment = String(process.env.VERCEL_ENV || "").trim().toLowerCase() || null;
  const environment = String(process.env.NEXT_PUBLIC_JUPR_ENV || "").trim().toLowerCase() || "production";
  const previewIsolationConfigured = truthy(process.env.NEXT_PUBLIC_JUPR_PREVIEW_ISOLATION);
  const previewAuthIsolationConfigured = truthy(
    process.env.NEXT_PUBLIC_JUPR_PREVIEW_AUTH_ISOLATION
  );
  const authOrigin = safeOrigin(
    process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_JUPR_SUPABASE_URL
  );
  const expectedStagingOrigin =
    safeOrigin(process.env.JUPR_STAGING_API_BASE_URL) ||
    "https://juprleagues-api-staging.fly.dev";
  const gitCommitSha = String(process.env.VERCEL_GIT_COMMIT_SHA || "").trim().toLowerCase() || null;
  const vercelDeploymentId = String(process.env.VERCEL_DEPLOYMENT_ID || "").trim() || null;
  const vercelDeploymentOrigin = safeOrigin(
    process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : undefined
  );

  return NextResponse.json(
    {
      environment,
      vercel_environment: vercelEnvironment,
      git_commit_sha: gitCommitSha,
      vercel_deployment_id: vercelDeploymentId,
      vercel_deployment_origin: vercelDeploymentOrigin,
      api_origin: apiOrigin,
      score_entry_visible: truthy(process.env.NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY),
      auth_origin: authOrigin,
      preview_isolation_configured: previewIsolationConfigured,
      preview_auth_isolation_configured: previewAuthIsolationConfigured,
      preview_isolation_active:
        environment === "staging" &&
        previewIsolationConfigured &&
        apiOrigin === expectedStagingOrigin,
      preview_auth_isolation_active:
        environment === "staging" &&
        previewAuthIsolationConfigured &&
        authOrigin === safeOrigin(process.env.NEXT_PUBLIC_STAGING_SUPABASE_URL)
    },
    { headers: { "Cache-Control": "no-store" } }
  );
}

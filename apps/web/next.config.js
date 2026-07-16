const isVercelPreview = process.env.VERCEL_ENV === "preview";
const defaultStagingApiBaseUrl = "https://juprleagues-api-staging.fly.dev";

function normalizeHttpsUrl(value, fallback) {
  const raw = String(value || fallback || "").trim().replace(/\/$/, "");
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`Invalid staging API URL: ${raw || "<empty>"}`);
  }
  if (parsed.protocol !== "https:") {
    throw new Error("Vercel preview API URL must use https://");
  }
  return parsed.origin;
}

const previewApiBaseUrl = isVercelPreview
  ? normalizeHttpsUrl(process.env.JUPR_STAGING_API_BASE_URL, defaultStagingApiBaseUrl)
  : null;
const previewSupabaseUrl = String(process.env.NEXT_PUBLIC_STAGING_SUPABASE_URL || "").trim();
const previewSupabaseAnonKey = String(
  process.env.NEXT_PUBLIC_STAGING_SUPABASE_ANON_KEY || ""
).trim();
const previewAuthIsolationConfigured = Boolean(previewSupabaseUrl && previewSupabaseAnonKey);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: isVercelPreview
    ? {
        JUPR_API_BASE_URL: previewApiBaseUrl,
        NEXT_PUBLIC_JUPR_API_BASE_URL: previewApiBaseUrl,
        NEXT_PUBLIC_JUPR_ENV: "staging",
        NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY: "1",
        NEXT_PUBLIC_JUPR_PREVIEW_ISOLATION: "1",
        NEXT_PUBLIC_JUPR_PREVIEW_AUTH_ISOLATION: previewAuthIsolationConfigured ? "1" : "0",
        ...(previewAuthIsolationConfigured
          ? {
              NEXT_PUBLIC_SUPABASE_URL: normalizeHttpsUrl(previewSupabaseUrl),
              NEXT_PUBLIC_SUPABASE_ANON_KEY: previewSupabaseAnonKey
            }
          : {})
      }
    : {}
};

module.exports = nextConfig;

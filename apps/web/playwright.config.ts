import { defineConfig, devices } from "@playwright/test";

const remoteBaseUrl = String(process.env.STAGING_WEB_BASE_URL || "").trim().replace(/\/$/, "");
const baseURL = remoteBaseUrl || "http://127.0.0.1:3000";
const bypassSecret = String(process.env.VERCEL_AUTOMATION_BYPASS_SECRET || "").trim();
const productionHosts = new Set([
  "pickleballclubsandwich.com",
  "www.pickleballclubsandwich.com",
  "juprleagues.com",
  "www.juprleagues.com",
  "jupr-rho.vercel.app",
  "jupr-pickleballattrespalapas1.vercel.app"
]);

if (remoteBaseUrl && productionHosts.has(new URL(remoteBaseUrl).hostname.toLowerCase())) {
  throw new Error(`Refusing to run the staging browser smoke against production host ${new URL(remoteBaseUrl).hostname}.`);
}

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 30_000,
  expect: { timeout: 8_000 },
  reporter: process.env.CI
    ? [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]]
    : "list",
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    extraHTTPHeaders: bypassSecret
      ? {
          "x-vercel-protection-bypass": bypassSecret,
          "x-vercel-set-bypass-cookie": "true"
        }
      : undefined,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure"
  },
  webServer: remoteBaseUrl
    ? undefined
    : {
        command: "npm run build && npm run start -- --hostname 127.0.0.1",
        url: "http://127.0.0.1:3000/api/environment",
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
        env: {
          ...process.env,
          VERCEL_ENV: "preview"
        }
      }
});

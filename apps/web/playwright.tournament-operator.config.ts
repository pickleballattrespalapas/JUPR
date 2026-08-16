import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  testMatch: "tournament-operator.local.spec.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:3107",
    screenshot: "only-on-failure",
    trace: "retain-on-failure"
  },
  webServer: {
    command: "npm run dev -- --hostname 127.0.0.1 --port 3107",
    url: "http://127.0.0.1:3107/api/environment",
    reuseExistingServer: false,
    timeout: 180_000,
    env: {
      ...process.env,
      JUPR_INTERACTION_TEST_HARNESS: "1",
      NEXT_PUBLIC_JUPR_API_BASE_URL: "http://127.0.0.1:3999",
      NEXT_PUBLIC_JUPR_ADMIN_CLUB_ID: "tres_palapas"
    }
  }
});

import {
  expect,
  type APIResponse,
  type BrowserContext,
  type Page
} from "@playwright/test";

export const clubSlug = String(process.env.JUPR_SMOKE_CLUB_SLUG || "tres-palapas").trim();
export const expectAuthIsolation = /^(1|true|yes|on)$/i.test(
  String(process.env.JUPR_EXPECT_PREVIEW_AUTH_ISOLATION || "")
);
export const expectedApiOrigin = String(
  process.env.JUPR_EXPECTED_STAGING_API_ORIGIN || "https://juprleagues-api-staging.fly.dev"
).trim().replace(/\/$/, "");
export const expectedAuthOrigin = String(process.env.JUPR_EXPECTED_STAGING_AUTH_ORIGIN || "")
  .trim()
  .replace(/\/$/, "");

const expectedStagingWebOrigin =
  "https://jupr-git-staging-pickleballattrespalapas1.vercel.app";
const remoteBaseUrl = String(process.env.STAGING_WEB_BASE_URL || "").trim().replace(/\/$/, "");
const attestedDeploymentOrigin = String(
  process.env.JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN || ""
).trim().replace(/\/$/, "");
const bypassSecret = String(process.env.VERCEL_AUTOMATION_BYPASS_SECRET || "").trim();
const immutableDeploymentHostname =
  /^[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}-pickleballattrespalapas1\.vercel\.app$/;
const vercelBypassOrigin = (() => {
  if (!remoteBaseUrl || !bypassSecret) return "";
  if (!attestedDeploymentOrigin) {
    if (remoteBaseUrl !== expectedStagingWebOrigin) {
      throw new Error("Refusing to send Vercel bypass credentials to a non-staging web origin.");
    }
    return expectedStagingWebOrigin;
  }
  let hostname = "";
  try {
    const parsed = new URL(remoteBaseUrl);
    if (parsed.origin !== remoteBaseUrl || parsed.protocol !== "https:") throw new Error();
    hostname = parsed.hostname;
  } catch {
    throw new Error("Refusing a non-canonical attested Vercel deployment origin.");
  }
  if (
    remoteBaseUrl === expectedStagingWebOrigin ||
    remoteBaseUrl !== attestedDeploymentOrigin ||
    !immutableDeploymentHostname.test(hostname)
  ) {
    throw new Error("Refusing to send Vercel bypass credentials outside the attested immutable deployment.");
  }
  return remoteBaseUrl;
})();

export type Surface = {
  name: string;
  path: string;
  expected: RegExp;
};

export async function bootstrapStagingContext(context: BrowserContext): Promise<void> {
  if (!vercelBypassOrigin || !bypassSecret) return;

  const bootstrapUrl = `${vercelBypassOrigin}/api/environment`;
  let bootstrap: APIResponse;
  try {
    bootstrap = await context.request.get(bootstrapUrl, {
      headers: {
        "x-vercel-protection-bypass": bypassSecret,
        "x-vercel-set-bypass-cookie": "true"
      },
      maxRedirects: 0,
      failOnStatusCode: false
    });
  } catch {
    throw new Error("Unable to establish the Vercel automation bypass cookie.");
  }

  try {
    expect(
      bootstrap.status(),
      "Vercel bypass-cookie bootstrap did not redirect"
    ).toBeGreaterThanOrEqual(300);
    expect(
      bootstrap.status(),
      "Vercel bypass-cookie bootstrap did not redirect"
    ).toBeLessThan(400);
    expect(
      bootstrap.headersArray().some(({ name }) => name.toLowerCase() === "set-cookie"),
      "Vercel bypass-cookie bootstrap did not issue a cookie"
    ).toBeTruthy();
  } finally {
    await bootstrap.dispose().catch(() => {});
  }

  let verification: APIResponse;
  try {
    verification = await context.request.get(bootstrapUrl, {
      maxRedirects: 0,
      failOnStatusCode: false
    });
  } catch {
    throw new Error("Unable to verify the Vercel automation bypass cookie.");
  }

  try {
    expect(verification.status(), "Vercel bypass cookie was not accepted").toBe(200);
    expect(verification.headers()["content-type"] || "").toContain("application/json");
  } finally {
    await verification.dispose().catch(() => {});
  }
}

export async function expectHealthySurface(page: Page, surface: Surface): Promise<void> {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const response = await page.goto(surface.path, { waitUntil: "domcontentloaded" });
  expect(response, `${surface.name} did not return a document response`).not.toBeNull();
  expect(response?.status(), `${surface.name} returned an error status`).toBeLessThan(400);
  await expect(page.locator("body")).toContainText(surface.expected);
  await expect(page.locator("body")).not.toHaveText("");
  await expect(page.locator("[data-nextjs-dialog], .nextjs-container-errors-pseudo-html")).toHaveCount(0);
  expect(pageErrors, `${surface.name} raised browser page errors`).toEqual([]);
  expect(consoleErrors, `${surface.name} emitted console errors`).toEqual([]);
}

import { expect, test } from "@playwright/test";
import { bootstrapStagingContext } from "./support/staging";

const clubSlug = String(process.env.STAGING_CLUB_SLUG || "tres-palapas").trim();
const tournamentId = String(process.env.STAGING_PARTNER_TOURNAMENT_ID || "").trim();
const registrationSlug = String(process.env.STAGING_PARTNER_REGISTRATION_SLUG || "").trim();
const requesterToken = String(process.env.STAGING_PARTNER_REQUESTER_EDIT_TOKEN || "").trim();
const requesterSelectionId = String(process.env.STAGING_PARTNER_REQUESTER_SELECTION_ID || "").trim();
const targetBoardEntryKey = String(process.env.STAGING_PARTNER_TARGET_BOARD_ENTRY_KEY || "").trim();
const apiBase = String(process.env.STAGING_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || "").trim().replace(/\/$/, "");

test.beforeEach(async ({ context }) => {
  await bootstrapStagingContext(context);
});

function boardQuery(extra: Record<string, string> = {}): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  for (const [key, value] of Object.entries(extra)) if (value) query.set(key, value);
  const encoded = query.toString();
  return encoded ? `?${encoded}` : "";
}

test("partner board renders the explicit privacy boundary", async ({ page }) => {
  await page.goto(`/clubs/${clubSlug}/tournament-partner-board${boardQuery()}`);

  await expect(page.getByText("Tournament Partner Board", { exact: true })).toBeVisible();
  await expect(page.getByText(/contact details are not exposed/i)).toBeVisible();
  await expect(page.locator('a[href^="mailto:"]')).toHaveCount(0);
  await expect(page.locator('a[href^="tel:"]')).toHaveCount(0);
});

test("disposable requester can create, review, and cancel a request", async ({ page, request }) => {
  test.skip(
    !apiBase || !tournamentId || !requesterToken || !requesterSelectionId || !targetBoardEntryKey,
    "Set the STAGING_PARTNER_* disposable fixture variables to run the mutating browser/API smoke."
  );

  const createResponse = await request.post(`${apiBase}/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-interest`, {
    data: {
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      edit_token: requesterToken,
      requester_selection_id: requesterSelectionId,
      board_entry_key: targetBoardEntryKey
    }
  });
  expect(createResponse.ok()).toBeTruthy();
  const created = await createResponse.json() as { partner_request_id: string; status: string };
  expect(created.partner_request_id).toBeTruthy();
  expect(created.status).toBe("PENDING");

  await page.goto(`/clubs/${clubSlug}/tournament-partner-board${boardQuery({ edit_token: requesterToken, partner_request_id: created.partner_request_id })}`);
  await expect(page.getByRole("heading", { name: "Your partner requests" })).toBeVisible();
  const cancel = page.getByRole("button", { name: "Cancel request" }).first();
  await expect(cancel).toBeVisible();
  await cancel.click();
  await page.getByRole("button", { name: "Confirm cancel request" }).first().click();
  await expect(page.getByText(/partner request cancelled/i)).toBeVisible();

  const cancelled = await request.get(`${apiBase}/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/pairing-requests`, {
    params: { edit_token: requesterToken, tournament_id: tournamentId, registration_slug: registrationSlug }
  });
  expect(cancelled.ok()).toBeTruthy();
  const reviewed = await cancelled.json() as { outgoing?: Array<{ id: string; status: string }> };
  expect(reviewed.outgoing?.find((row) => row.id === created.partner_request_id)?.status).toBe("CANCELLED");
});

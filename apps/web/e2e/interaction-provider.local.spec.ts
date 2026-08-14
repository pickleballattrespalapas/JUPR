import { expect, test } from "@playwright/test";

test.describe("root interaction provider", () => {
  test.skip(
    Boolean(String(process.env.STAGING_WEB_BASE_URL || "").trim()),
    "The interaction foundation harness is available only on the local test server."
  );

  test.beforeEach(async ({ page }) => {
    await page.goto("/__interaction-provider-test");
  });

  test("keeps success visible after the originating consumer unmounts", async ({ page }) => {
    await page.getByRole("button", { name: "Remove record" }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Remove record" }).click();

    await expect(page.locator("#record-gone")).toBeVisible();
    await expect(page.getByRole("button", { name: "Remove record" })).toHaveCount(0);
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("heading", { name: "Record removed" })).toBeVisible();
    await expect(dialog).toContainText("The authoritative record no longer exists.");
    await expect(dialog.getByRole("heading", { name: "Record removed" })).toHaveCount(1);

    await dialog.getByRole("button", { name: "OK" }).click();
    await expect(dialog).toHaveCount(0);
    await expect.poll(() => page.evaluate(() => document.activeElement?.tagName)).toBe("MAIN");
  });

  test("accepts only one interaction synchronously", async ({ page }) => {
    await page.getByRole("button", { name: "Exercise one-action lock" }).click();

    await expect(page.getByTestId("lock-result")).toHaveText("true:false");
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("heading", { name: "First action" })).toBeVisible();
    await expect(dialog.getByRole("heading", { name: "Second action" })).toHaveCount(0);
    await dialog.getByRole("button", { name: "Cancel first" }).click();
  });

  test("focusTargetId takes precedence over the connected trigger", async ({ page }) => {
    const trigger = page.getByRole("button", { name: "Check focus precedence" });
    await trigger.click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Complete focus check" }).click();
    await expect(dialog.getByRole("heading", { name: "Focus priority confirmed" })).toBeVisible();
    await expect(dialog.getByRole("heading", { name: "Focus priority confirmed" })).toHaveCount(1);

    await dialog.getByRole("button", { name: "OK" }).click();
    await expect(page.locator("#explicit-focus-target")).toBeFocused();
    await expect(trigger).not.toBeFocused();
  });

  test("falls back to main when the connected trigger becomes disabled", async ({ page }) => {
    const trigger = page.getByRole("button", { name: "Disable trigger after success" });
    await trigger.click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Complete disabled-trigger check" }).click();

    await expect(trigger).toBeDisabled();
    await expect(dialog.getByRole("heading", { name: "Disabled-trigger fallback confirmed" })).toBeVisible();
    await dialog.getByRole("button", { name: "OK" }).click();

    await expect.poll(() => page.evaluate(() => document.activeElement?.tagName)).toBe("MAIN");
    await expect(trigger).not.toBeFocused();
  });

  test("form dialog falls back to main when its connected trigger becomes disabled", async ({ page }) => {
    const trigger = page.getByRole("button", { name: "Open form focus check" });
    await trigger.click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Complete form focus check" }).click();

    await expect(trigger).toBeDisabled();
    await expect(dialog.getByRole("heading", { name: "Form disabled-trigger fallback confirmed" })).toBeVisible();
    await dialog.getByRole("button", { name: "Done" }).click();

    await expect.poll(() => page.evaluate(() => document.activeElement?.tagName)).toBe("MAIN");
    await expect(trigger).not.toBeFocused();
  });

  test("form dialog success target takes precedence over its connected trigger", async ({ page }) => {
    const trigger = page.getByRole("button", { name: "Open explicit form focus check" });
    await trigger.click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Complete explicit form focus check" }).click();

    await expect(dialog.getByRole("heading", { name: "Form focus priority confirmed" })).toBeVisible();
    await dialog.getByRole("button", { name: "Done" }).click();

    await expect(page.locator("#form-explicit-focus-target")).toBeFocused();
    await expect(trigger).not.toBeFocused();
  });

  test("keeps an unsaved division draft through equivalent parent rerenders", async ({ page }) => {
    await page.getByRole("button", { name: "Open add division retention check" }).click();
    const dialog = page.getByRole("dialog");
    const name = dialog.getByLabel("Division name");

    await name.clear();
    await name.pressSequentially("DO NOT RESET", { delay: 10 });
    const rendersBefore = Number(
      (await page.getByTestId("division-parent-render-count").textContent()) || 0
    );
    await expect
      .poll(async () =>
        Number(
          (await page.getByTestId("division-parent-render-count").textContent()) || 0
        )
      )
      .toBeGreaterThan(rendersBefore + 2);
    await expect(name).toHaveValue("DO NOT RESET");

    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(
      dialog.getByRole("heading", { name: "Discard unsaved changes?" })
    ).toBeVisible();
    await dialog.getByRole("button", { name: "Discard changes" }).click();
    await expect(dialog).toHaveCount(0);

    await page.getByRole("button", { name: "Open add division retention check" }).click();
    await expect(page.getByRole("dialog").getByLabel("Division name")).toHaveValue(
      "Default division name"
    );
    await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click();
  });

  test("keeps an edit draft through rerenders and refreshes it on the next open", async ({ page }) => {
    await page.getByRole("button", { name: "Open edit division retention check" }).click();
    const dialog = page.getByRole("dialog");
    const name = dialog.getByLabel("Division name");

    await expect(name).toHaveValue("Default division name");
    await name.clear();
    await name.pressSequentially("  Interaction Test Division Edited  ", {
      delay: 10
    });
    const rendersBefore = Number(
      (await page.getByTestId("division-parent-render-count").textContent()) || 0
    );
    await expect
      .poll(async () =>
        Number(
          (await page.getByTestId("division-parent-render-count").textContent()) || 0
        )
      )
      .toBeGreaterThan(rendersBefore + 2);
    await expect(name).toHaveValue("  Interaction Test Division Edited  ");

    await dialog.getByRole("button", { name: "Save division" }).click();
    await expect(
      dialog.getByRole("heading", { name: "Division retained" })
    ).toBeVisible();
    await expect(page.getByTestId("division-submitted-names")).toHaveText(
      "Interaction Test Division Edited|Public composite label"
    );
    await dialog.getByRole("button", { name: "Done" }).click();

    await page.getByRole("button", { name: "Open edit division retention check" }).click();
    await expect(page.getByRole("dialog").getByLabel("Division name")).toHaveValue(
      "Interaction Test Division Edited"
    );

    await page.getByRole("dialog").getByRole("button", { name: "Cancel" }).click();
    await page
      .getByRole("button", { name: "Update authoritative division source" })
      .click();
    await page.getByRole("button", { name: "Open edit division retention check" }).click();
    await expect(page.getByRole("dialog").getByLabel("Division name")).toHaveValue(
      "Updated authoritative division"
    );
  });
});

# Weekly reports and member newsletter sections

In **Admin → Weekly Recap**, choose **Last completed week**, **This week so far**, or **Last 7 days**, then generate a draft. Custom date ranges remain available. Club dates use America/Mazatlan, including Sunday-night/Monday boundaries.

The expanded report includes named generator sessions, Moneyball settlement using the frozen expectations model, separate RR standings and playoff winners, division-level tournament podiums from tournament closeout receipts, and published interclub results. Pending generator submissions appear in the private coverage review, never in public results. Counts come from saved match seats, so late arrivals and departed players are included without counting generator games a second time.

Player highlights use explicit minimum-game rules. Top Performer consistently means wins minus losses, with opponent strength as its first tiebreaker and a four-game minimum. Above Expectations uses pre-game rating snapshots for at least four rated games; unrated games and games lacking those snapshots cannot manufacture an expected result. Season races use configured league award definitions, show eligibility progress, and snapshot standings through the report end date. New achievements require a triggering game in the report period, rather than the date a badge was inserted.

The five-step builder follows **Setup → Content → Personalize → Preview → Publish & share**. Choose the whole club or one league before generating. League scope filters games, totals, candidates, and season races; excluding an event card in Content only changes the stories shown, not the scope of the totals.

Turn each section on or off, rename it, move it earlier or later, and choose brief or full detail. Select individual statistics, event results, season races, achievements, and new members, with separate selections for the compact member email. Empty sections disappear. Personalize the headline, opening message, club notes, spotlight choices, upcoming events, and links. Add optional announcements, coaching tips, recognition, sponsor messages, and photos using public HTTPS image URLs with alternative text.

Club stories have individual recap and email inclusion switches. Choose **Simple story**, **Photo feature**, **Highlighted note**, or **Promotion with button**. Photos can show the whole image or crop to a wide banner, with a caption and credit. **Featured event** starts a registration promotion; its button label and destination are editable. Each card has an inline preview. Hiding a story keeps it in the draft for later editing and removes it from every public edition and export.

In Content, open **Arrange your email** to move email sections and individual items independently of the website. **Use website order for email** restores the website ordering while retaining your email selections. Email content remains a subset of the full recap. Website and email order are both remembered when you save a layout.

Drafts save automatically after editing pauses. A failed save stays visible and requires a retry or an explicit reload; optimistic row versions reject concurrent edits. Reopening a draft restores its wizard step. **Refresh results** preserves editorial work and stable item selections; unavailable selections are omitted rather than replaced with a different event or winner. Newly discovered events are not silently added to an explicit item selection.

Published recaps have a separate frozen snapshot. **Edit draft revision** allows changes and refreshed results while the existing edition remains public. **Publish recap** replaces that snapshot only after confirmation. **Unpublish recap** removes it from the public list. Review coverage warnings and the live preview before publishing.

Save named layouts and choose a club default for future editions. Layouts remember section order, headings, detail, email visibility, statistics, and email spotlight categories. They do not copy dated event selections, player choices, or editorial stories into a new week. Templates are versioned per club independently from drafts. The existing one-edition-per-club/start-date rule remains in place.

Open **Manage saved layouts** in Setup to rename or delete a layout, change the default, or return to the standard recap. Deleting the current default restores Standard for new editions. These actions leave existing recaps and the current editor unchanged. Conflicting edits require **Reload saved layouts** before retrying; reloading layouts preserves the draft. Names must be unique within the club.

## Email workflow

1. Save the draft edits and review the full report.
2. Choose **Compact newsletter section** or **Full weekly report** under **Add to your member email**.
3. **Copy email HTML** into CourtReserve's HTML/source editor, inside the newsletter's content area. Alternatively download the email section and provide that file when preparing the weekly member email in the Tres 2027 project.
4. Keep the newsletter's own greeting, Joe signature, and CourtReserve footer. The report section supplies none of those a second time.

Exports use the established Tres palette (#123e37 green, #f8f3e9 sand, #17665c teal, #cb794f terracotta), Arial/Helvetica, and a fluid table capped at 620 pixels. Inline styles and table layouts do not require JavaScript in the email. Registration links remain clickable in HTML. Compact exports default to three events, two spotlight categories, two season races, and three achievements. The builder can select the exact items instead. A published-report button leads to the complete edition. Draft exports are explicitly labeled and omit the unpublished public-report link.

**Download HTML preview** produces a standalone document; **Download email section** produces the embeddable table. **Download PNG** produces a static image from the same email layout even while the preview is collapsed. **Download styled PDF** preserves the photos, card styling, captions, and clickable links. It uses the selected compact/full content size and avoids splitting cards that fit on one page. Published builder editions also offer this PDF on the public page. The separate **Text-only PDF** retains searchable text and photo captions; browser printing/save-as-PDF remains available. All outputs honor section and item exclusions; website, print, and full email use the full selection, while compact email uses its explicit subset.

PNG and styled PDF exports run in the browser, loading their rendering libraries only on demand. Public photo hosts must allow cross-origin downloads; a blocked or missing photo stops the export with a visible error instead of silently omitting it. Use Print-friendly view for those hosts or unusually long editions. Photos are supplied by HTTPS URL; this build does not add a photo upload service.

Exports never send email. Staging retains `JUPR_EMAIL_MODE=dry_run`. Migration `20261019053313_weekly_recap_builder.sql` adds the published snapshot and server-only club layout preferences. Existing published editions are backfilled and retain their legacy layout until an admin publishes a revision through the builder. Public JSON exposes only the selected edition, without the private candidate catalog or coverage review. The legacy Streamlit public page and PDF reader also use the frozen snapshot and selected edition.

Checks: Python recap service/domain tests, shared Python/TypeScript projection fixture, component checks, and a loopback browser rehearsal using the real Next editor and FastAPI recap routes with synthetic source results and identity. Run the rehearsal with `node apps/web/tests/recap-builder-browser.cjs` after installing Python API dependencies and Playwright Chromium. It never connects to external services.

`apps/web/tests/fixtures/weekly-report.json` is explicitly illustrative data for rendering checks; its numbers and names are not actual Tres results.

## Results links and league reports

Generator cards link directly to shared standings only when viewing has been enabled and the session is readable. Sessions without a public results page keep their verified summary on the card. Match Explorer and an unrelated league's results are never used as fallback destinations. Interclub cards open the published season directly.

Social round robins show the event winner and record using the canonical wins, point difference, and points scored rules. Numerical ties stay labeled as tied pending the club's decision; a recorded decision is honored only while its result fingerprint remains valid. Saved legacy social events use their recorded event snapshot, with saved match seats as a fallback.

League cards in the report include **Weekly report · Week N** links for every league week played in the report period. These links also appear in published email exports, including compact exports. The public league directory has a **Weekly report** shortcut on each card. An unqualified weekly link opens the latest week with results, while explicit week selections remain available.

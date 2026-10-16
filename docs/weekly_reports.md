# Weekly reports and member newsletter sections

In **Admin → Weekly Recap**, choose **Last completed week**, **This week so far**, or **Last 7 days**, then generate a draft. Custom date ranges remain available. Club dates use America/Mazatlan, including Sunday-night/Monday boundaries.

The expanded report includes named generator sessions, Moneyball settlement using the frozen expectations model, separate RR standings and playoff winners, division-level tournament podiums from tournament closeout receipts, and published interclub results. Pending generator submissions appear in the private coverage review, never in public results. Counts come from saved match seats, so late arrivals and departed players are included without counting generator games a second time.

Player highlights use explicit minimum-game rules. Top Performer consistently means wins minus losses, with opponent strength as its first tiebreaker and a four-game minimum. Above Expectations uses pre-game rating snapshots for at least four rated games; unrated games and games lacking those snapshots cannot manufacture an expected result. Season races use configured league award definitions, show eligibility progress, and snapshot standings through the report end date. New achievements require a triggering game in the report period, rather than the date a badge was inserted.

Edit the headline, opening message, club notes, spotlight choices, and registration links. **Refresh results** preserves these edits; unavailable spotlight choices are omitted rather than silently replaced. Published recaps must be returned to draft before changing their results. Review coverage warnings before publishing.

## Email workflow

1. Save the draft edits and review the full report.
2. Choose **Compact newsletter section** or **Full weekly report** under **Add to your member email**.
3. **Copy email HTML** into CourtReserve's HTML/source editor, inside the newsletter's content area. Alternatively download the email section and provide that file when preparing the weekly member email in the Tres 2027 project.
4. Keep the newsletter's own greeting, Joe signature, and CourtReserve footer. The report section supplies none of those a second time.

Exports use the established Tres palette (#123e37 green, #f8f3e9 sand, #17665c teal, #cb794f terracotta), Arial/Helvetica, and a fluid table capped at 620 pixels. Inline styles and table layouts do not require JavaScript in the email. Registration links remain clickable in HTML. Compact exports feature three events, two spotlight categories, two season races, and three achievements; a published-report button leads to the complete edition. Draft exports are explicitly labeled and omit the unpublished public-report link.

**Download HTML preview** produces a standalone document; **Download email section** produces the embeddable table. **Download PNG** produces a static image from the same email layout even while the preview is collapsed. Use HTML to retain clickable links. Existing browser printing and PDF exports remain available.

Exports never send email. Staging retains `JUPR_EMAIL_MODE=dry_run` and never targets production. No schema migration is required; additions are stored in the existing recap JSON snapshots, and old published editions retain their legacy layout until regenerated.

`apps/web/tests/fixtures/weekly-report.json` is explicitly illustrative data for rendering checks; its numbers and names are not actual Tres results.

## Results links and league reports

Generator cards link directly to shared standings only when viewing has been enabled and the session is readable. Sessions without a public results page keep their verified summary on the card. Match Explorer and an unrelated league's results are never used as fallback destinations. Interclub cards open the published season directly.

Social round robins show the event winner and record using the canonical wins, point difference, and points scored rules. Numerical ties stay labeled as tied pending the club's decision; a recorded decision is honored only while its result fingerprint remains valid. Saved legacy social events use their recorded event snapshot, with saved match seats as a fallback.

League cards in the report include **Weekly report · Week N** links for every league week played in the report period. These links also appear in published email exports, including compact exports. The public league directory has a **Weekly report** shortcut on each card. An unqualified weekly link opens the latest week with results, while explicit week selections remain available.

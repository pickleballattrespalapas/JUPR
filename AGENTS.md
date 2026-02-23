# Repository Agent Instructions

These instructions apply to the entire repository.

## Branch and workflow constraints
- Work **ONLY** on branch `rebuild`.
- Run tests before finishing any task.

## Dependency policy
- Do **not** add new runtime dependencies unless strictly necessary.
- If a runtime dependency is added, justify why in documentation.

## Recap Composer MVP implementation constraints
- Branding-first output only: canonical HTML plus PDF export generated from HTML.
- Do **not** implement WhatsApp plaintext recap output.
- Admin must pick effective dates (`start` and `end`) for every recap.
- Supported recap levels: `weekly`, `monthly`, `yearly`.
- Weekly preset definition: the last full Monday–Sunday week in `America/Mexico_City`, represented as the half-open interval `[Mon 00:00, next Mon 00:00)` (exclusive end).
- Featured Upcoming Event (CTA) is required at publish time (admin must pick one before publish).
- Featured Past Event is optional and may only be chosen if it occurred within `[report_start, report_end)`.
- Monthly recaps must include a Member of the Month block, required at publish time.
- Tournaments must be included in recaps:
  - tournaments within the reporting window
  - upcoming tournaments within the configured lookahead window
- Publish and visibility flow:
  - Drafts are private and admin-only.
  - Published recaps are public by default with visibility override: `public` / `unlisted` / `private`.
  - `public`: appears in archive.
  - `unlisted`: accessible by direct link only.
  - `private`: admin-only.
- Published recaps are stable snapshots and must not change after publish, even if underlying source data changes.

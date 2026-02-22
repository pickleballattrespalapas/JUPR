# Recap Composer MVP Scope (Implementation Guidance)

## Purpose
This document defines implementation guidance for the Recap Composer MVP. It captures product rules and data expectations for building recap composition, preview, publishing, and public consumption flows while keeping published recaps immutable.

## Guardrails
- Branch: all implementation work for this MVP is constrained to `rebuild`.
- Output format is branding-first: canonical HTML is the source of truth, with PDF export generated from that HTML.
- WhatsApp plaintext output is explicitly out of scope.
- No new runtime dependencies unless strictly necessary; any additions must be justified.

## Recap Levels and Date Semantics
- Recap levels:
  - `weekly`
  - `monthly`
  - `yearly`
- Admin must always select effective report dates (`report_start`, `report_end`) for every recap.
- Weekly preset semantics (timezone-aware):
  - Timezone: `America/Mexico_City`
  - Last full week Monday through Sunday
  - Stored/computed as half-open interval: `[Mon 00:00, next Mon 00:00)`
  - End is exclusive.

## Screens

### 1) Dashboard
- List recaps by status (`draft`, `published`) and visibility (`public`, `unlisted`, `private`).
- Quick metadata: level, report range, publish date, visibility, featured events selected.
- Actions:
  - create new recap
  - edit draft
  - open preview/publish
  - open published snapshot

### 2) Composer
- Inputs:
  - recap level (`weekly`/`monthly`/`yearly`)
  - report start/end dates (admin-selected)
  - optional preset assistance for weekly range (last full week rule)
- Content picks:
  - required featured upcoming event (CTA) for publish
  - optional featured past event from events in `[report_start, report_end)`
  - tournaments in reporting window
  - upcoming tournaments in configured lookahead window
  - monthly: Member of the Month candidate (required before publish)
- Output assembly remains HTML-first.

### 3) Preview & Publish
- Preview canonical HTML and generated PDF.
- Validate publish-time requirements:
  - featured upcoming event selected
  - monthly recap includes Member of the Month
- Visibility selector at publish:
  - `public` (default)
  - `unlisted`
  - `private`
- Publish creates immutable snapshot payload (frozen content).

### 4) Public Recap Page
- Render published snapshot only.
- Respect visibility:
  - `public`: accessible and discoverable
  - `unlisted`: accessible by direct URL only
  - `private`: not public
- Must not rehydrate from mutable source data after publish.

### 5) Archive Page
- Shows only `public` published recaps.
- Sorting/filtering by level/date.
- Links to each recap snapshot page.

## Data Model Guidance

Suggested MVP entity for stored recap snapshot (`recaps`):

- `id`
- `level` (`weekly` | `monthly` | `yearly`)
- `status` (`draft` | `published`)
- `visibility` (`public` | `unlisted` | `private`)
- `report_start` (datetime)
- `report_end` (datetime, exclusive)
- `timezone` (default `America/Mexico_City`)
- `effective_label` (optional human label)

Feature picks and required blocks:
- `featured_upcoming_event_id` (required at publish)
- `featured_past_event_id` (optional; must be within report window)
- `member_of_month_member_id` (required for monthly at publish)

Tournament blocks:
- `tournaments_in_window` (snapshot list of tournaments in `[report_start, report_end)`)
- `tournaments_upcoming` (snapshot list for lookahead horizon)
- `tournament_lookahead_days` (or equivalent config reference copied into snapshot)

Snapshot output fields:
- `html_snapshot` (canonical rendered HTML)
- `pdf_snapshot_url` or `pdf_snapshot_blob_ref`
- `snapshot_version`
- `published_at`
- `created_by_admin_id`
- `updated_at` (draft edits only)

Immutability rule:
- When `status = published`, recap content fields (`html_snapshot`, selected feature IDs, tournaments blocks, Member of the Month block) are immutable.
- Post-publish source data changes must not alter previously published recap content.

## Section Templates by Recap Level

All recap levels include:
1. Header/branding
2. Date range summary
3. Featured Upcoming Event (CTA, required for publish)
4. Tournaments section:
   - in-window tournaments
   - upcoming tournaments (lookahead)
5. Footer/meta

### Weekly template
- Intro summary
- Featured Upcoming Event (required)
- Optional Featured Past Event (if selected and valid in range)
- Tournaments in report week
- Upcoming tournaments (lookahead)
- Closing

### Monthly template
- Intro summary
- Member of the Month block (required at publish)
- Featured Upcoming Event (required)
- Optional Featured Past Event
- Tournaments in month window
- Upcoming tournaments (lookahead)
- Closing

### Yearly template
- Year summary
- Featured Upcoming Event (required)
- Optional Featured Past Event
- Tournaments in yearly window
- Upcoming tournaments (lookahead)
- Closing

## Publish and Visibility Rules
- Draft recaps are admin-only.
- Publishing defaults visibility to `public` unless explicitly overridden.
- Visibility meanings:
  - `public`: listed in archive and accessible directly
  - `unlisted`: excluded from archive, accessible by direct link
  - `private`: admin-only even after publish
- Publish action writes immutable snapshot for rendering.

## Non-goals for this MVP document
- No implementation of messaging channel variants (e.g., WhatsApp plaintext).
- No runtime dependency additions unless justified by implementation need.

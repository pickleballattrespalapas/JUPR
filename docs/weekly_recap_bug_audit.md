# Weekly Recap Bug Audit

## Scope
- Feature audited: **Weekly Recap** only.
- Branch: `rebuild`.
- Constraint followed: bug-fix stabilization only (no feature expansion, no format redesign).

## 1) Entrypoints and dependency map

### UI pages
- Public/Admin recap display page: `jupr_app/ui/pages/weekly_recap.py` (`render(ctx)`).
- Weekly Recap Admin page: `jupr_app/ui/pages/weekly_recap_admin.py` (`render(ctx)`).
- Route wiring: `streamlit_app.py` maps `weekly_recap` and `weekly_recap_admin` page keys.

### Service layer and recap assembly
- Core recap payload generation:
  - `compute_weekly_recap(...)` in `jupr_app/domain/recaps/weekly_recap.py`.
  - `_compute_weekly_recap_payload(...)` in same file.
- Spotlight candidates for admin overrides:
  - `get_spotlight_candidates(...)` in same file.

### Data fetchers / query paths
- Match loading:
  - `_load_week_matches(...)` reads from in-memory `ctx.df_matches` first, then Supabase `matches` table fallback.
- Derived stats and sections:
  - `_compute_stats(...)`, `_build_numbers(...)`, `_compute_new_faces(...)`, `_build_around_club(...)`.
- Challenge ladder summary:
  - `_fetch_challenge_ladder_week_summary(...)`.

### Rendering / templates
- HTML generation and section rendering:
  - `build_weekly_recap_html(...)` in `jupr_app/ui/components/weekly_recap_layout.py`.
- Streamlit embedding:
  - `render_weekly_recap(...)` in same file.
- PDF generation path (admin):
  - `_pdf_for_recap(...)` + download button flow in `jupr_app/ui/pages/weekly_recap_admin.py`.

### Background job / CLI / scheduled task
- No standalone Weekly Recap CLI command or scheduler entrypoint was found in this repository during this audit.

## 2) Failure reproduction and findings

## Commands run for reproduction and verification
- `pytest -q tests -k weekly_recap`
- `pytest -q tests/test_weekly_recap_regression.py tests/test_weekly_recap_bounds.py tests/test_weekly_recap_delta.py tests/test_weekly_recap_spotlight.py tests/test_weekly_recap_challenge_ladder.py tests/test_weekly_recap_layout_challenge_ladder.py`
- Static trace/search:
  - `rg -n "weekly_recap|compute_weekly_recap|weekly_recap_admin|weekly_recaps" jupr_app streamlit_app.py tests -S`

### Bug A — Missing tenant scoping in local match path
- **Symptom:** Weekly recap could include matches from other clubs when `ctx.df_matches` contains multi-club rows.
- **Repro (pre-fix logic):** `_load_week_matches(...)` filtered local matches by date only, not by `club_id`.
- **Impact:** Inaccurate recap statistics and spotlight data for the selected club.
- **Category:** Incorrect data pull / missing club scoping.

### Bug B — PDF export hard failure risk (WeasyPrint import/runtime)
- **Symptom:** Admin page could crash in environments lacking WeasyPrint runtime dependencies.
- **Repro path:** `_pdf_for_recap(...)` imported WeasyPrint directly and admin render flow called it unguarded.
- **Impact:** Runtime exception blocks recap admin page actions and prevents PDF download flow from degrading gracefully.
- **Category:** Runtime reliability failure.

### Empty dataset handling status
- Validated that recap computation and HTML rendering complete without crash when no weekly matches are present.
- Ensured output remains structurally valid/non-empty HTML with zeroed number tiles.

## 3) Fixes applied

1. Added `club_id` filtering to `_load_week_matches(...)` for local DataFrame path before date filtering.
2. Hardened `_pdf_for_recap(...)` to raise controlled runtime error when WeasyPrint import is unavailable.
3. Wrapped admin PDF generation in a guarded `try/except` and show warning fallback instead of crashing the page.

## 4) Files modified
- `jupr_app/domain/recaps/weekly_recap.py`
- `jupr_app/ui/pages/weekly_recap_admin.py`
- `tests/test_weekly_recap_regression.py`
- `docs/weekly_recap_bug_audit.md`

## 5) What was NOT changed (explicit)
- No new recap sections were added.
- No output format restructuring was introduced.
- No newsletter content/tone logic was changed.
- No performance optimization/refactor unrelated to correctness/reliability was introduced.
- No new configuration knobs were added.

## 6) Minimal regression tests added
- Added tests covering:
  - Local DataFrame tenant scoping in `_load_week_matches(...)`.
  - End-to-end recap payload + HTML non-empty output with valid input data.
  - End-to-end recap payload + HTML safe handling with empty dataset.

## 7) Remaining technical debt (not fixed in this pass)
- Admin preview uses session-state render mode flags that are currently set/reset in page code; behavior works but remains loosely coupled and lightly exercised by tests.
- PDF generation still depends on deployment runtime packages for full capability; this pass adds graceful degradation but does not bundle system libs.
- Supabase query resilience in some helper paths relies on broad exception handling patterns that could benefit from tighter typed error branches in future hardening work.

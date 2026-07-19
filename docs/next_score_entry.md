# Next Score Entry

The canonical minimal score-entry route is `/clubs/{clubSlug}/admin/score-entry`. It is a guarded one-match fallback to the full Match Uploader, not an independent rating implementation.

## Readiness contract

The browser form is rendered only when both layers agree it is ready:

- Next has `NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1`.
- FastAPI `GET /admin/clubs/{club_id}/score-entry/status` reports its write flag enabled and `SUPABASE_SERVICE_ROLE_KEY` configured.

If either layer is off or the readiness request fails, the page shows Match Uploader and Streamlit fallback links instead of a write form.

## Write contract

`POST /admin/clubs/{club_id}/matches/batch` requires Supabase JWT authentication, `enter_scores` permission, the FastAPI flag, and the server-only service role. Despite the legacy endpoint name, Score Entry accepts exactly one match with four distinct players, non-negative whole-number scores, a non-zero result, and no tie. Larger batches belong in Match Uploader.

FastAPI calls the existing Python `submit_match_batch` domain service and returns `match_write_committed=true`, rating feedback, and Match Log/Match Uploader/Replay History recovery links. If the browser loses the response, staff must verify Match Log before retrying.

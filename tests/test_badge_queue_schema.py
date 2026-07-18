from pathlib import Path
import re


MIGRATION = Path(
    "supabase/migrations/20260718141016_badge_eval_queue_atomic_club_claim.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _claim_function(sql: str) -> str:
    match = re.search(
        r"create or replace function public\.claim_badge_eval_queue_job\(.*?\$function\$;",
        sql,
        flags=re.DOTALL,
    )
    assert match is not None, "missing atomic badge queue claim function"
    return match.group(0)


def test_badge_queue_safety_uses_canonical_supabase_migration():
    assert MIGRATION.is_file()
    assert MIGRATION.parent.as_posix() == "supabase/migrations"


def test_badge_queue_deduplication_is_club_scoped_and_postgrest_compatible():
    sql = _sql()

    assert "drop index if exists public.badge_eval_queue_event_match_key" in sql
    assert "drop index if exists public.badge_eval_queue_event_match_uidx" in sql
    assert "create unique index if not exists badge_eval_queue_club_event_match_key" in sql
    assert "on public.badge_eval_queue (club_id, event_type, match_id)" in sql
    assert "group by queue.club_id, queue.event_type, queue.match_id" in sql
    assert "having count(*) > 1" in sql


def test_badge_queue_claim_is_atomic_club_scoped_and_fail_closed():
    sql = _sql()
    claim = _claim_function(sql)

    assert "returns setof public.badge_eval_queue" in claim
    assert "security invoker" in claim
    assert "set search_path = ''" in claim
    assert "queue.club_id = pg_catalog.btrim(p_club_id)" in claim
    assert "queue.status = 'pending'" in claim
    assert "order by queue.created_at asc, queue.id asc" in claim
    assert "for update skip locked" in claim
    assert "status = 'processing'" in claim
    assert "attempts = queue.attempts + 1" in claim
    assert "returning queue.*" in claim

    assert "alter table public.badge_eval_queue enable row level security" in sql
    assert (
        "revoke all on table public.badge_eval_queue from public, anon, authenticated"
        in sql
    )
    assert (
        "revoke all on function public.claim_badge_eval_queue_job(text)\n"
        "  from public, anon, authenticated"
        in sql
    )
    assert (
        "grant execute on function public.claim_badge_eval_queue_job(text)\n"
        "  to service_role"
        in sql
    )
    assert (
        "grant select, insert, update on table public.badge_eval_queue to service_role"
        in sql
    )
    assert "notify pgrst, 'reload schema'" in sql

-- Final cleanup for deprecated Badge V2 schema after full V3 rollout.
-- Safe to run repeatedly.

-- 1) Ensure legacy V2 rules table is fully removed.
drop table if exists public.badge_rules_v2;

-- 2) Remove legacy uniqueness artifacts that still include context_type.
--    Canonical uniqueness is (club_id, player_id, badge_id, context_id).
alter table if exists public.player_badges
    drop constraint if exists player_badges_unique_context_type,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_type_context_id_key,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key;

-- Drop any lingering unique indexes on player_badges that include context_type.
do $$
declare
    idx record;
begin
    for idx in
        select indexname
        from pg_indexes
        where schemaname = 'public'
          and tablename = 'player_badges'
          and indexdef ilike 'create unique index%'
          and indexdef ilike '%(club_id, player_id, badge_id, context_type, context_id)%'
    loop
        execute format('drop index if exists public.%I', idx.indexname);
    end loop;
end $$;

-- Canonical player_badges uniqueness contract:
-- (club_id, player_id, badge_id, context_id)
-- context_id encodes scope; context_type is informational only.

with ranked as (
    select
        id,
        row_number() over (
            partition by club_id, player_id, badge_id, context_id
            order by earned_at asc nulls last, id asc
        ) as rn
    from player_badges
)
delete from player_badges
where id in (select id from ranked where rn > 1);

do $$
declare
    constraint_name text;
begin
    for constraint_name in
        select con.conname
        from pg_constraint con
        join pg_class rel on rel.oid = con.conrelid
        join pg_namespace nsp on nsp.oid = rel.relnamespace
        join unnest(con.conkey) with ordinality as cols(attnum, ord) on true
        join pg_attribute att on att.attrelid = rel.oid and att.attnum = cols.attnum
        where con.contype = 'u'
          and nsp.nspname = 'public'
          and rel.relname = 'player_badges'
        group by con.conname
        having bool_or(att.attname = 'context_type')
    loop
        execute format('alter table public.player_badges drop constraint if exists %I', constraint_name);
    end loop;

    execute 'alter table public.player_badges drop constraint if exists player_badges_unique_context';
    execute 'alter table public.player_badges drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key';
    execute 'alter table public.player_badges drop constraint if exists player_badges_unique_context_type';

    if not exists (
        select 1
        from pg_constraint con
        join pg_class rel on rel.oid = con.conrelid
        join pg_namespace nsp on nsp.oid = rel.relnamespace
        where con.contype = 'u'
          and nsp.nspname = 'public'
          and rel.relname = 'player_badges'
          and con.conname = 'player_badges_unique_context'
    ) then
        alter table public.player_badges
            add constraint player_badges_unique_context
            unique (club_id, player_id, badge_id, context_id);
    end if;
end $$;

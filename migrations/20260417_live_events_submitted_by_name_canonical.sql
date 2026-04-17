-- Canonical submitter field for Club Social is live_events.submitted_by_name.
-- Forward migration:
--   1) Ensure submitted_by_name exists.
--   2) Backfill from legacy submitted_by when present.
--   3) Keep submitted_by_name synchronized when legacy submitted_by is written during transition.

alter table if exists public.live_events
    add column if not exists submitted_by_name text null;

-- Backfill canonical column from legacy column where available.
do $$
begin
    if exists (
        select 1
        from information_schema.columns
        where table_schema = 'public'
          and table_name = 'live_events'
          and column_name = 'submitted_by'
    ) then
        execute $sql$
            update public.live_events
               set submitted_by_name = coalesce(nullif(submitted_by_name, ''), submitted_by)
             where coalesce(nullif(submitted_by_name, ''), '') = ''
               and coalesce(nullif(submitted_by, ''), '') <> ''
        $sql$;
    end if;
end;
$$;

-- Transition trigger: if some clients still write submitted_by, mirror it into canonical submitted_by_name.
create or replace function public.live_events_sync_submitted_by_name()
returns trigger
language plpgsql
as $$
begin
    if new.submitted_by_name is null or btrim(new.submitted_by_name) = '' then
        if exists (
            select 1
            from information_schema.columns
            where table_schema = 'public'
              and table_name = 'live_events'
              and column_name = 'submitted_by'
        ) then
            execute 'select ($1).submitted_by' into new.submitted_by_name using new;
        end if;
    end if;
    return new;
end;
$$;

drop trigger if exists live_events_sync_submitted_by_name on public.live_events;
create trigger live_events_sync_submitted_by_name
before insert or update on public.live_events
for each row
execute function public.live_events_sync_submitted_by_name();

-- Rollback SQL:
--   drop trigger if exists live_events_sync_submitted_by_name on public.live_events;
--   drop function if exists public.live_events_sync_submitted_by_name();
--   alter table if exists public.live_events drop column if exists submitted_by_name;

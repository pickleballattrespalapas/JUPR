do $$
begin
    if not exists (select 1 from pg_type where typname = 'badge_state') then
        create type badge_state as enum ('live', 'frozen', 'deprecated');
    end if;
end $$;

alter table if exists badges
    add column if not exists state badge_state not null default 'live',
    add column if not exists state_changed_at timestamptz not null default now(),
    add column if not exists state_change_reason text;

update badges
set
    state = coalesce(state, 'live'),
    state_changed_at = coalesce(state_changed_at, now());

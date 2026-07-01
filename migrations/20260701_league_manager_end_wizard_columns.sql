alter table if exists public.leagues_metadata
    add column if not exists status text,
    add column if not exists started_at timestamptz,
    add column if not exists ended_at timestamptz,
    add column if not exists ended_by text,
    add column if not exists end_awards jsonb,
    add column if not exists schedule_config jsonb,
    add column if not exists court_board_defaults jsonb,
    add column if not exists rules_config jsonb,
    add column if not exists awards_config jsonb;

alter table if exists public.leagues_metadata
    alter column status set default 'draft';

update public.leagues_metadata
set status = case
    when ended_at is not null then 'ended'
    when is_active is true then 'active'
    else 'draft'
end
where status is null;

alter table if exists public.badges
    add column if not exists eval_triggers jsonb not null default '["match_recorded","match_updated"]'::jsonb;

update public.badges
set eval_triggers = '["match_recorded","match_updated"]'::jsonb
where eval_triggers is null;

-- Editorial choices only. Match results and ratings remain authoritative.
create table public.league_weekly_highlights (
  club_id text not null,
  league_name text not null,
  week_num integer not null check (week_num > 0),
  mode text not null check (mode in ('automatic', 'curated')),
  selected_ids jsonb not null default '[]'::jsonb
    check (jsonb_typeof(selected_ids) = 'array' and jsonb_array_length(selected_ids) <= 6),
  revision bigint not null default 1 check (revision > 0),
  updated_at timestamptz not null default now(),
  updated_by text not null,
  primary key (club_id, league_name, week_num),
  foreign key (club_id, league_name) references public.leagues_metadata (club_id, league_name)
    on update cascade on delete cascade,
  check (mode = 'curated' or selected_ids = '[]'::jsonb)
);
alter table public.league_weekly_highlights enable row level security;
revoke all on public.league_weekly_highlights from public, anon, authenticated;
grant all on public.league_weekly_highlights to service_role;

create function public.save_league_weekly_highlights(
  p_club_id text, p_league_name text, p_week_num integer, p_mode text,
  p_selected_ids jsonb, p_expected_revision bigint, p_actor_email text, p_actor_role text
) returns jsonb
language plpgsql security invoker set search_path = '' as $$
declare
  previous public.league_weekly_highlights%rowtype;
  saved public.league_weekly_highlights%rowtype;
begin
  if p_actor_role is null or p_actor_role not in ('super_admin', 'club_owner', 'administrator') then
    raise exception 'Only club administrators can select weekly highlights.' using errcode = '42501';
  end if;
  if nullif(trim(p_actor_email), '') is null or p_week_num is null or p_week_num < 1
     or p_expected_revision is null or p_expected_revision < 0
     or p_mode is null or p_mode not in ('automatic', 'curated')
     or p_selected_ids is null or jsonb_typeof(p_selected_ids) <> 'array' then
    raise exception 'Invalid weekly highlights selection.' using errcode = '22023';
  end if;
  if jsonb_array_length(p_selected_ids) > 6
     or (p_mode = 'automatic' and p_selected_ids <> '[]'::jsonb)
     or exists (select 1 from jsonb_array_elements(p_selected_ids) v where jsonb_typeof(v) <> 'string')
     or exists (select 1 from jsonb_array_elements_text(p_selected_ids) v where length(v) > 160 or length(v) = 0)
     or (select count(distinct v) from jsonb_array_elements_text(p_selected_ids) v) <> jsonb_array_length(p_selected_ids) then
    raise exception 'Choose up to 6 distinct highlights.' using errcode = '22023';
  end if;
  -- Covers concurrent first saves as well as updates without locking other weeks.
  perform pg_advisory_xact_lock(hashtextextended(jsonb_build_array(p_club_id, p_league_name, p_week_num)::text, 0));
  select * into previous from public.league_weekly_highlights
    where club_id = p_club_id and league_name = p_league_name and week_num = p_week_num for update;
  -- A retry after a lost response is harmless, including its audit entry.
  if previous.revision is not null and previous.mode = p_mode and previous.selected_ids = p_selected_ids then
    return to_jsonb(previous) - 'updated_by';
  end if;
  if coalesce(previous.revision, 0) <> p_expected_revision then
    raise exception 'Weekly highlights changed; reload before saving.' using errcode = '40001';
  end if;
  insert into public.league_weekly_highlights (club_id, league_name, week_num, mode, selected_ids, revision, updated_by)
    values (p_club_id, p_league_name, p_week_num, p_mode, p_selected_ids, coalesce(previous.revision, 0) + 1, p_actor_email)
    on conflict (club_id, league_name, week_num) do update set
      mode = excluded.mode, selected_ids = excluded.selected_ids, revision = excluded.revision,
      updated_at = now(), updated_by = excluded.updated_by
    returning * into saved;
  insert into public.admin_activity_log
    (club_id, actor_email, actor_role, action_type, entity_type, entity_id, before_json, after_json, source_page, flagged_for_review)
    values (p_club_id, p_actor_email, p_actor_role, 'league_weekly_highlights_save', 'league_weekly_highlights',
            jsonb_build_array(p_league_name, p_week_num)::text, to_jsonb(previous), to_jsonb(saved), 'next_weekly_highlights', false);
  return to_jsonb(saved) - 'updated_by';
end;
$$;
revoke all on function public.save_league_weekly_highlights(text,text,integer,text,jsonb,bigint,text,text) from public, anon, authenticated;
grant execute on function public.save_league_weekly_highlights(text,text,integer,text,jsonb,bigint,text,text) to service_role;

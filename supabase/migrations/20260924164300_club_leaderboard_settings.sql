begin;

create table public.club_leaderboard_settings (
  club_id text primary key references public.clubs(id),
  revision integer not null default 0 check (revision >= 0),
  draft jsonb not null default '{}'::jsonb check (jsonb_typeof(draft) = 'object'),
  published jsonb check (jsonb_typeof(published) = 'object'),
  published_at timestamptz,
  updated_at timestamptz not null default now()
);
create table public.club_leaderboard_settings_audit (
  id bigint generated always as identity primary key,
  club_id text not null references public.clubs(id),
  actor_id uuid not null,
  action text not null check (action in ('save','publish','discard')),
  before_state jsonb not null,
  after_state jsonb not null,
  created_at timestamptz not null default now()
);
create index club_leaderboard_settings_audit_club_time_idx on public.club_leaderboard_settings_audit(club_id, created_at desc);
alter table public.club_leaderboard_settings enable row level security;
alter table public.club_leaderboard_settings_audit enable row level security;
revoke all on public.club_leaderboard_settings, public.club_leaderboard_settings_audit from public, anon, authenticated;
grant select, insert, update on public.club_leaderboard_settings to service_role;
grant select, insert on public.club_leaderboard_settings_audit to service_role;
grant usage, select on sequence public.club_leaderboard_settings_audit_id_seq to service_role;

-- Definer access is limited to this service-only RPC so actor verification can
-- read auth.users without granting the API role access to private user records.
create function public.save_club_leaderboard_settings(
  p_club_id text, p_actor_id uuid, p_actor_email text,
  p_expected_revision integer, p_action text, p_settings jsonb default null
) returns jsonb language plpgsql security definer set search_path = '' as $$
declare
  prior public.club_leaderboard_settings;
  saved public.club_leaderboard_settings;
begin
  -- Serialize with staff revocations and repeat authorization inside the transaction.
  perform pg_advisory_xact_lock(hashtextextended('pcs-staff:' || p_club_id, 0));
  if not exists (
    select 1 from public.admin_role_assignments a join auth.users u on u.id = p_actor_id
    where a.club_id = p_club_id and a.email = lower(trim(p_actor_email))
      and lower(u.email) = lower(trim(p_actor_email)) and u.email_confirmed_at is not null
      and (a.user_id is null or a.user_id = p_actor_id)
      and a.role in ('super_admin','club_owner','administrator')
      and a.revoked_at is null and (a.expires_at is null or a.expires_at > now())
  ) then raise exception 'Administrator access required' using errcode = '42501'; end if;
  if p_expected_revision is null or p_expected_revision < 0
    or p_action is null or p_action not in ('save','publish','discard') then
    raise exception 'Invalid settings action' using errcode = '22023';
  end if;
  if p_action = 'save' and (p_settings is null or jsonb_typeof(p_settings) <> 'object'
      or octet_length(p_settings::text) > 100000) then
    raise exception 'Invalid leaderboard settings' using errcode = '22023';
  end if;
  insert into public.club_leaderboard_settings(club_id) values(p_club_id) on conflict do nothing;
  select * into prior from public.club_leaderboard_settings where club_id = p_club_id for update;
  if prior.revision <> p_expected_revision then
    raise exception 'Settings changed; reload before retrying' using errcode = '40001';
  end if;
  update public.club_leaderboard_settings set
    draft = case when p_action = 'save' then p_settings
                 when p_action = 'discard' then coalesce(published, '{}'::jsonb) else draft end,
    published = case when p_action = 'publish' then draft else published end,
    published_at = case when p_action = 'publish' then now() else published_at end,
    revision = revision + 1, updated_at = now()
  where club_id = p_club_id returning * into saved;
  insert into public.club_leaderboard_settings_audit(club_id, actor_id, action, before_state, after_state)
    values(p_club_id, p_actor_id, p_action, to_jsonb(prior), to_jsonb(saved));
  return to_jsonb(saved);
end $$;
revoke all on function public.save_club_leaderboard_settings(text,uuid,text,integer,text,jsonb) from public, anon, authenticated;
grant execute on function public.save_club_leaderboard_settings(text,uuid,text,integer,text,jsonb) to service_role;
notify pgrst, 'reload schema';
commit;

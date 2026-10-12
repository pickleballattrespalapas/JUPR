-- Support-only breadcrumbs, not registrations and not evidence of identity.
create table public.tournament_registration_attempt_events (
  id uuid primary key,
  attempt_id uuid not null,
  club_id text not null references public.clubs(id) on delete cascade,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  source text not null check (source in ('BROWSER','SERVER')),
  event text not null check (event in ('PROGRESS','CLIENT_REJECTED','LEFT_FORM','SUBMITTING','SERVER_REJECTED','PROFILE_READY','SAVED')),
  stage text not null check (stage in ('contact','profile','events','review','submit','edit')),
  flow text not null check (flow in ('new','edit')),
  name text not null default '' check (length(name) <= 160),
  email text not null default '' check (length(email) <= 320),
  message text not null default '' check (length(message) <= 500),
  error_code text not null default '' check (length(error_code) <= 60),
  selections jsonb not null default '[]' check (jsonb_typeof(selections) = 'array' and jsonb_array_length(selections) <= 8 and length(selections::text) <= 4000),
  registration_id text null,
  created_at timestamptz not null default now(),
  check (source = 'SERVER' or (registration_id is null and event in ('PROGRESS','CLIENT_REJECTED','LEFT_FORM')))
);
create index registration_attempt_scope_time on public.tournament_registration_attempt_events (club_id,tournament_id,created_at desc);
create index registration_attempt_email on public.tournament_registration_attempt_events (club_id,tournament_id,email,created_at desc);
create index registration_attempt_timeline on public.tournament_registration_attempt_events (club_id,tournament_id,attempt_id,created_at);
create index registration_attempt_retention on public.tournament_registration_attempt_events (created_at);
create index registration_attempt_club_time on public.tournament_registration_attempt_events (club_id,created_at);
alter table public.tournament_registration_attempt_events enable row level security;
revoke all on public.tournament_registration_attempt_events from public, anon, authenticated;
grant select,insert,delete on public.tournament_registration_attempt_events to service_role;

create function public.record_tournament_registration_attempt(p_event jsonb)
returns boolean language plpgsql security invoker set search_path = '' as $$
declare
  v_club text := p_event->>'club_id';
  v_tournament uuid := (p_event->>'tournament_id')::uuid;
  v_attempt uuid := (p_event->>'attempt_id')::uuid;
begin
  if not exists (select 1 from public.tournaments where id=v_tournament and club_id=v_club) then
    return false;
  end if;
  -- Serialize the small quota check to keep limits durable across API workers.
  perform pg_advisory_xact_lock(hashtextextended('registration-support:'||v_club,0));
  if p_event->>'event' = 'SAVED' then
    if exists (select 1 from public.tournament_registration_attempt_events where club_id=v_club
      and tournament_id=v_tournament and attempt_id=v_attempt and source='SERVER' and event='SAVED') then
      return false;
    end if;
  elsif (select count(*) from public.tournament_registration_attempt_events where club_id=v_club
    and tournament_id=v_tournament and attempt_id=v_attempt) >= 100
    or (select count(*) from public.tournament_registration_attempt_events where club_id=v_club
    and created_at >= now()-interval '1 day') >= 24000 then
    return false;
  end if;
  if p_event->>'source' = 'BROWSER' then
    if (select count(*) from public.tournament_registration_attempt_events
        where club_id=v_club and tournament_id=v_tournament and attempt_id=v_attempt and source='BROWSER') >= 80
       or (select count(*) from public.tournament_registration_attempt_events
           where club_id=v_club and created_at >= now()-interval '1 day' and source='BROWSER') >= 12000 then
      return false;
    end if;
  end if;
  insert into public.tournament_registration_attempt_events
    (id,attempt_id,club_id,tournament_id,source,event,stage,flow,name,email,message,error_code,selections,registration_id)
  values ((p_event->>'id')::uuid,v_attempt,v_club,v_tournament,p_event->>'source',p_event->>'event',p_event->>'stage',
          p_event->>'flow',p_event->>'name',p_event->>'email',p_event->>'message',p_event->>'error_code',
          p_event->'selections',p_event->>'registration_id') on conflict (id) do nothing;
  return true;
end;
$$;
revoke all on function public.record_tournament_registration_attempt(jsonb) from public,anon,authenticated;
grant execute on function public.record_tournament_registration_attempt(jsonb) to service_role;

-- Supabase supplies pg_cron. Do not retain contact breadcrumbs indefinitely.
select cron.schedule('registration-support-retention','17 4 * * *',
  $$delete from public.tournament_registration_attempt_events where created_at < now()-interval '30 days'$$);

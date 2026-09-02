-- Tournament Live awards compare every reviewed podium row by id/version.
-- Order 27 added that application contract but did not add the corresponding
-- tournament_podium.updated_at column when the table already existed.

do $$
begin
  if pg_catalog.to_regclass('public.tournament_podium') is null then
    raise exception 'Tournament podium must exist before podium row versions are installed';
  end if;
end
$$;

alter table public.tournament_podium
  add column if not exists updated_at timestamptz not null
  default pg_catalog.timezone('utc', pg_catalog.now());

-- Repair a partially applied/manual schema without weakening the final
-- not-null version contract.
update public.tournament_podium
   set updated_at = coalesce(
     updated_at,
     created_at,
     pg_catalog.timezone('utc', pg_catalog.now())
   )
 where updated_at is null;

alter table public.tournament_podium
  alter column updated_at set default pg_catalog.timezone('utc', pg_catalog.now()),
  alter column updated_at set not null;

create or replace function public.advance_tournament_podium_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  new.updated_at := greatest(
    pg_catalog.clock_timestamp(),
    old.updated_at + interval '1 microsecond'
  );
  return new;
end;
$$;

drop trigger if exists trg_tournament_podium_advance_updated_at
  on public.tournament_podium;
create trigger trg_tournament_podium_advance_updated_at
before update on public.tournament_podium
for each row execute function public.advance_tournament_podium_updated_at();

revoke all on function public.advance_tournament_podium_updated_at()
  from public, anon, authenticated;
grant execute on function public.advance_tournament_podium_updated_at()
  to service_role;

comment on column public.tournament_podium.updated_at is
  'Server-maintained row version used by guarded Tournament Live podium review and awards.';
comment on function public.advance_tournament_podium_updated_at() is
  'Advances each podium row version monotonically before an update.';

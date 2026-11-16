-- Keep canonical names for staff, CourtReserve matching, and match attribution.
alter table public.players add column if not exists public_alias text;
alter table public.players add constraint players_public_alias_valid
  check (public_alias is null or (length(btrim(public_alias)) between 1 and 160
    and public_alias !~ '[<>[:cntrl:]]'));
comment on column public.players.public_alias is
  'Optional public presentation name. Never replaces the canonical name or player ID.';
create index if not exists players_public_alias_club_idx on public.players (club_id, id)
  where public_alias is not null;
notify pgrst, 'reload schema';

alter table if exists public.matches
  drop column if exists created_at;

create index if not exists idx_matches_active_club_date
  on public.matches (club_id, date desc)
  where deleted_at is null;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'clubs_slug_format_check' and conrelid = 'public.clubs'::regclass) then
    alter table public.clubs add constraint clubs_slug_format_check check (slug ~ '^[a-z0-9-]+$');
  end if;
  if not exists (select 1 from pg_constraint where conname = 'clubs_features_json_object_check' and conrelid = 'public.clubs'::regclass) then
    alter table public.clubs add constraint clubs_features_json_object_check check (jsonb_typeof(features_json) = 'object');
  end if;
  if not exists (select 1 from pg_constraint where conname = 'leagues_metadata_event_tags_object_check' and conrelid = 'public.leagues_metadata'::regclass) then
    alter table public.leagues_metadata add constraint leagues_metadata_event_tags_object_check check (jsonb_typeof(event_tags) = 'object');
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournaments_date_order_check' and conrelid = 'public.tournaments'::regclass) then
    alter table public.tournaments add constraint tournaments_date_order_check check (start_date is null or end_date is null or end_date >= start_date);
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournaments_event_tags_object_check' and conrelid = 'public.tournaments'::regclass) then
    alter table public.tournaments add constraint tournaments_event_tags_object_check check (jsonb_typeof(event_tags) = 'object');
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournament_registration_selections_partner_mode_check' and conrelid = 'public.tournament_registration_selections'::regclass) then
    alter table public.tournament_registration_selections add constraint tournament_registration_selections_partner_mode_check check (partner_mode in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER'));
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournament_registration_selections_registration_fk' and conrelid = 'public.tournament_registration_selections'::regclass) then
    alter table public.tournament_registration_selections add constraint tournament_registration_selections_registration_fk foreign key (registration_id) references public.tournament_registrations(id) on delete cascade;
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournament_registration_selections_day_fk' and conrelid = 'public.tournament_registration_selections'::regclass) then
    alter table public.tournament_registration_selections add constraint tournament_registration_selections_day_fk foreign key (registration_day_id) references public.tournament_registration_days(id) on delete cascade;
  end if;
  if not exists (select 1 from pg_constraint where conname = 'tournament_registration_selections_event_option_fk' and conrelid = 'public.tournament_registration_selections'::regclass) then
    alter table public.tournament_registration_selections add constraint tournament_registration_selections_event_option_fk foreign key (event_option_id) references public.tournament_event_options(id) on delete cascade;
  end if;
end
$$;

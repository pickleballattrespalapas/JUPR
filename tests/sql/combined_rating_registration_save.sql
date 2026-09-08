-- Run against staging or an isolated database with the canonical migrations.
-- Uses the API's service_role and real registration RPC/trigger chain.
-- All fixtures are synthetic and rolled back; no API/email is invoked.
-- Example: psql "$SUPABASE_TEST_DATABASE_URL" -v ON_ERROR_STOP=1 -f this-file.sql
begin;
set local statement_timeout = '30s';
set local role service_role;

do $test$
declare
  v_tournament uuid := gen_random_uuid();
  v_suffix text := replace(gen_random_uuid()::text, '-', '');
  v_club text;
  v_day text;
  v_event text;
  v_registration text;
  v_selection text;
  v_partner_player integer;
  v_partner_name text;
  v_partner_registration text;
  v_partner_selection text;
  v_legacy_primary text;
  v_legacy_partner text;
  v_legacy_selection text;
  v_result jsonb;
begin
  select id::text into strict v_club from public.clubs order by id limit 1;
  v_day := 'day_regression_' || v_suffix;
  v_event := 'event_regression_' || v_suffix;
  v_registration := 'reg_regression_' || v_suffix;
  v_selection := 'sel_regression_' || v_suffix;
  v_partner_name := 'SQL Regression Partner ' || v_suffix;
  insert into public.tournaments(id, club_id, name, status, team_count)
    values(v_tournament, v_club, 'Disposable registration regression', 'DRAFT', 8);
  insert into public.tournament_registration_days(id, tournament_id, sort_order, label)
    values(v_day, v_tournament::text, 0, 'Regression day');
  insert into public.tournament_event_options(
    id, tournament_id, registration_day_id, sort_order, label, event_type,
    partner_required, eligibility_mode, combined_rating_cap
  ) values (
    v_event, v_tournament::text, v_day, 0, 'Combined below 9.0', 'MIXED_DOUBLES',
    true, 'COMBINED_RATING_CAP', 9
  );

  -- Reuse an existing partner profile while creating a new primary profile.
  insert into public.players(club_id, name, rating, starting_rating, wins,
    losses, matches_played, active)
    values(v_club, v_partner_name, 1360, 1360, 0, 0, 0, true)
    returning id into v_partner_player;
  v_result := public.create_tournament_registration_canonical_v1(
    jsonb_build_object(
      'id', v_registration, 'tournament_id', v_tournament::text,
      'submitted_at', now(), 'updated_at', now(), 'status', 'confirmed',
      'payment_status', 'unpaid', 'display_name', 'SQL Regression Primary ' || v_suffix,
      'email', v_suffix || '@example.invalid', 'doubles_skill', 4.5,
      'age', 40, 'gender', 'Men'
    ),
    jsonb_build_array(jsonb_build_object(
      'id', v_selection, 'registration_day_id', v_day, 'event_option_id', v_event,
      'partner_mode', 'HAS_PARTNER', 'partner_name', v_partner_name,
      'partner_email', 'partner-' || v_suffix || '@example.invalid',
      'partner_age', 40, 'partner_gender', 'Women', 'partner_skill', 3.4,
      'show_on_partner_board', false, 'sort_order', 0,
      'created_at', now(), 'updated_at', now()
    ))
  );
  if (v_result ->> 'ok') is distinct from 'true' then
    raise exception 'Canonical public registration did not succeed';
  end if;
  if (select count(*) from public.tournament_registrations
      where tournament_id = v_tournament::text) <> 2
     or (select count(*) from public.tournament_registration_selections
      where tournament_id = v_tournament::text) <> 2
     or (select count(*) from public.tournament_registration_team_links
      where tournament_id = v_tournament and status = 'ADMIN_CONFIRMED') <> 1
     or (select count(*) from public.tournament_registration_team_members
      where tournament_id = v_tournament and status = 'ACTIVE') <> 2 then
    raise exception 'Registration must create two entries and one canonical team';
  end if;
  select id into strict v_partner_registration from public.tournament_registrations
    where tournament_id = v_tournament::text and player_id = v_partner_player;
  select id into strict v_partner_selection from public.tournament_registration_selections
    where registration_id = v_partner_registration and event_option_id = v_event;
  if (select count(*) from public.players
      where club_id = v_club and name = v_partner_name) <> 1 then
    raise exception 'Existing partner profile was duplicated';
  end if;
  if (select count(*) from public.tournament_rating_eligibility_reviews
      where tournament_id = v_tournament and review_phase = 'INITIAL'
        and state = 'ELIGIBLE' and combined_rating = 7.9
        and player_rating_source = 'PCS_LINKED'
        and partner_rating_source = 'PCS_LINKED') <> 2 then
    raise exception 'Both canonical team members must receive matching rating reviews';
  end if;

  -- A repeat refresh exercises the UPDATE trigger on rows without draw_id.
  perform public.refresh_initial_combined_rating_review_v1(v_selection, 'sql-regression');
  if (select count(*) from public.tournament_rating_eligibility_reviews
      where tournament_id = v_tournament) <> 2 then
    raise exception 'Repeated initial review created duplicate reviews';
  end if;
  if not exists (select 1 from public.tournament_team_audit_events
      where tournament_id = v_tournament and actor = 'sql-regression'
        and action = 'rating_eligibility_initial_refreshed') then
    raise exception 'Initial rating review audit is missing';
  end if;

  -- The cap stays exclusive: equality must remain ineligible.
  update public.tournament_event_options set combined_rating_cap = 7.9 where id = v_event;
  perform public.refresh_initial_combined_rating_review_v1(v_selection, 'sql-regression');
  perform public.refresh_initial_combined_rating_review_v1(v_partner_selection, 'sql-regression');
  if (select count(*) from public.tournament_rating_eligibility_reviews
      where tournament_id = v_tournament and state = 'INELIGIBLE') <> 2 then
    raise exception 'A combined rating equal to the cap must be ineligible';
  end if;
  update public.tournament_event_options set combined_rating_cap = 9 where id = v_event;

  -- Legacy/unlinked entries resolve their partner by email and use organizer
  -- verification. This covers submitted_at plus both UUID verification queries.
  v_legacy_primary := 'reg_legacy_primary_' || v_suffix;
  v_legacy_partner := 'reg_legacy_partner_' || v_suffix;
  v_legacy_selection := 'sel_legacy_' || v_suffix;
  insert into public.tournament_registrations(id, tournament_id, display_name, email)
    values
      (v_legacy_primary, v_tournament::text, 'SQL Legacy Primary', 'legacy-' || v_suffix || '@example.invalid'),
      (v_legacy_partner, v_tournament::text, 'SQL Legacy Partner', 'legacy-partner-' || v_suffix || '@example.invalid');
  insert into public.tournament_registration_selections(
    id, tournament_id, registration_id, registration_day_id, event_option_id,
    partner_mode, partner_email
  ) values (
    v_legacy_selection, v_tournament::text, v_legacy_primary, v_day, v_event,
    'NONE', 'legacy-partner-' || v_suffix || '@example.invalid'
  );
  if not exists (select 1 from public.tournament_rating_eligibility_reviews
      where selection_id = v_legacy_selection and state = 'REVIEW_REQUIRED'
        and partner_registration_id = v_legacy_partner) then
    raise exception 'Unlinked email-matched partners must require rating review';
  end if;
  insert into public.tournament_rating_verifications(
    tournament_id, event_option_id, registration_id, rating, verified_by
  ) values
    (v_tournament, v_event, v_legacy_primary, 4.0, 'sql-regression'),
    (v_tournament, v_event, v_legacy_partner, 3.5, 'sql-regression');
  perform public.refresh_initial_combined_rating_review_v1(v_legacy_selection, 'sql-regression');
  if not exists (select 1 from public.tournament_rating_eligibility_reviews
      where selection_id = v_legacy_selection and state = 'ELIGIBLE'
        and combined_rating = 7.5 and player_rating_source = 'ORGANIZER_VERIFIED'
        and partner_rating_source = 'ORGANIZER_VERIFIED') then
    raise exception 'Organizer-verified ratings were not applied to both partners';
  end if;
  update public.tournament_rating_verifications set rating = 5.5
    where tournament_id = v_tournament and registration_id = v_legacy_primary;
  perform public.refresh_initial_combined_rating_review_v1(v_legacy_selection, 'sql-regression');
  if not exists (select 1 from public.tournament_rating_eligibility_reviews
      where selection_id = v_legacy_selection and state = 'INELIGIBLE'
        and combined_rating = 9) then
    raise exception 'Updating organizer verification must update eligibility';
  end if;

  update public.tournament_registration_selections
    set partner_mode = 'NEEDS_PARTNER', partner_email = null where id = v_legacy_selection;
  if not exists (select 1 from public.tournament_rating_eligibility_reviews
      where selection_id = v_legacy_selection and state = 'PROVISIONAL_NEEDS_PARTNER'
        and combined_rating is null) then
    raise exception 'An entry needing a partner must remain provisional';
  end if;
end;
$test$;

rollback;

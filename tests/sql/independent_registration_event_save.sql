-- Staging/isolated database only. Exercise the real edit RPC and triggers.
-- Synthetic fixtures and all changes roll back; no API or email is invoked.
begin;
set local statement_timeout = '30s';
set local role service_role;
do $test$
declare
  club text;
  tournament uuid;
  suffix text;
  scenario text;
  primary_id text;
  other_id text;
  men_id text;
  mixed_id text;
  other_selection text;
  event_men text;
  event_mixed text;
  day_id text;
  request_id text;
  link_id text;
  registration public.tournament_registrations%rowtype;
  selections jsonb;
  versions jsonb;
  before_men jsonb;
  before_requests jsonb;
  before_links jsonb;
  before_members jsonb;
  result jsonb;
begin
  select id::text into strict club from public.clubs order by id limit 1;
  foreach scenario in array array['PENDING', 'CANCELLED_PARTNER'] loop
    tournament := gen_random_uuid();
    suffix := replace(tournament::text, '-', '');
    primary_id := 'reg_primary_' || suffix;
    other_id := 'reg_other_' || suffix;
    men_id := 'sel_men_' || suffix;
    mixed_id := 'sel_mixed_' || suffix;
    other_selection := 'sel_other_' || suffix;
    event_men := 'event_men_' || suffix;
    event_mixed := 'event_mixed_' || suffix;
    day_id := 'day_' || suffix;
    request_id := 'request_' || suffix;
    link_id := 'link_' || suffix;
    insert into public.tournaments(id, club_id, name, status, team_count)
      values(tournament, club, 'Disposable independent event regression', 'DRAFT', 8);
    insert into public.tournament_registration_days(id, tournament_id, sort_order, label)
      values(day_id, tournament::text, 0, 'Regression day');
    insert into public.tournament_event_options(id, tournament_id, registration_day_id, sort_order, label, event_type, partner_required)
      values(event_men, tournament::text, day_id, 0, 'Men Open', 'GENDER_DOUBLES', true),
            (event_mixed, tournament::text, day_id, 1, 'Mixed Split Age', 'MIXED_DOUBLES', true);
    insert into public.tournament_registrations(id, tournament_id, display_name, email, status, age, gender, doubles_skill)
      values(primary_id, tournament::text, 'Regression Primary', 'primary-' || suffix || '@example.invalid', 'confirmed', 38, 'Men', 5),
            (other_id, tournament::text, 'Regression Other', 'other-' || suffix || '@example.invalid',
             case when scenario = 'CANCELLED_PARTNER' then 'cancelled' else 'confirmed' end, 42, 'Men', 5);
    insert into public.tournament_registration_selections(id, tournament_id, registration_id, registration_day_id, event_option_id, partner_mode, sort_order)
      values(men_id, tournament::text, primary_id, day_id, event_men, 'HAS_PARTNER', 0),
            (mixed_id, tournament::text, primary_id, day_id, event_mixed, 'NEEDS_PARTNER', 1),
            (other_selection, tournament::text, other_id, day_id, event_men, 'HAS_PARTNER', 0);
    insert into public.tournament_registration_partner_requests(id, tournament_id, event_option_id, requester_selection_id,
      requester_registration_id, target_selection_id, target_registration_id, status, source)
      values(request_id, tournament, event_men, men_id, primary_id, other_selection, other_id,
             case when scenario = 'PENDING' then 'PENDING' else 'ACCEPTED' end, 'NEEDS_PARTNER_LIST');
    if scenario = 'CANCELLED_PARTNER' then
      insert into public.tournament_registration_team_links(id, tournament_id, event_option_id, registration1_id,
        registration2_id, selection1_id, selection2_id, status, accepted_request_id)
        values(link_id, tournament, event_men, other_id, primary_id, other_selection, men_id, 'CONFIRMED', request_id);
      insert into public.tournament_registration_team_members(id, team_link_id, tournament_id, event_option_id,
        selection_id, registration_id, player_order, status)
        values('member1_' || suffix, link_id, tournament, event_men, other_selection, other_id, 1, 'ACTIVE'),
              ('member2_' || suffix, link_id, tournament, event_men, men_id, primary_id, 2, 'ACTIVE');
    end if;
    select * into strict registration from public.tournament_registrations where id = primary_id;
    select to_jsonb(s) - 'updated_at' into before_men from public.tournament_registration_selections s where id = men_id;
    select coalesce(jsonb_agg(to_jsonb(r) order by id), '[]') into before_requests from public.tournament_registration_partner_requests r where event_option_id = event_men;
    select coalesce(jsonb_agg(to_jsonb(l) order by id), '[]') into before_links from public.tournament_registration_team_links l where event_option_id = event_men;
    select coalesce(jsonb_agg(to_jsonb(m) order by id), '[]') into before_members from public.tournament_registration_team_members m where event_option_id = event_men;
    select jsonb_agg(jsonb_build_object('id', id, 'updated_at', updated_at)) into versions
      from public.tournament_registration_selections where registration_id = primary_id;
    select jsonb_agg((select jsonb_object_agg(key, value) from jsonb_each(to_jsonb(s)) where key = any(array[
      'id','tournament_id','registration_id','registration_day_id','event_option_id','partner_mode','partner_name',
      'partner_email','partner_phone','partner_dupr_id','partner_skill','partner_age','partner_gender','partner_note',
      'show_on_partner_board','sort_order','created_at','updated_at'])) ||
      case when s.id = mixed_id then jsonb_build_object('partner_mode','HAS_PARTNER','partner_name','Regression Mixed Partner',
        'partner_email','mixed-' || suffix || '@example.invalid','partner_skill',4.651203499999999,'partner_age',50,'partner_gender','Women')
      else '{}'::jsonb end order by s.sort_order) into selections
      from public.tournament_registration_selections s where registration_id = primary_id;
    result := public.server_update_public_tournament_registration_edit(
      p_tournament_id => tournament::text, p_registration_id => primary_id,
      p_expected_updated_at => registration.updated_at, p_expected_selection_versions => versions,
      p_registration_patch => jsonb_build_object('display_name',registration.display_name,'age',registration.age,'gender',registration.gender,
        'doubles_skill',registration.doubles_skill,'wants_partner_board_contact',false), p_selections => selections);
    if (result ->> 'ok') is distinct from 'true' then
      raise exception 'Independent save failed in %: %', scenario, result;
    end if;
    if (select to_jsonb(s) - 'updated_at' from public.tournament_registration_selections s where id = men_id) is distinct from before_men
      or (select coalesce(jsonb_agg(to_jsonb(r) order by id), '[]') from public.tournament_registration_partner_requests r where event_option_id = event_men) is distinct from before_requests
      or (select coalesce(jsonb_agg(to_jsonb(l) order by id), '[]') from public.tournament_registration_team_links l where event_option_id = event_men) is distinct from before_links
      or (select coalesce(jsonb_agg(to_jsonb(m) order by id), '[]') from public.tournament_registration_team_members m where event_option_id = event_men) is distinct from before_members then
      raise exception 'Unrelated partner arrangement changed in %', scenario;
    end if;
    if (select partner_name from public.tournament_registration_selections where id = mixed_id) <> 'Regression Mixed Partner'
      or (select count(*) from public.tournament_registration_team_links where event_option_id = event_mixed and status = 'ADMIN_CONFIRMED') <> 1 then
      raise exception 'Mixed partner was not saved as a canonical team';
    end if;
  end loop;
end $test$;
select 'pending and cancelled-partner arrangements preserved; mixed saves passed' as result;
rollback;

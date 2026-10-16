-- Move confirmed doubles teams together without replacing registrations,
-- selections, partner acceptance, or commerce records. All writes are one
-- transaction and remain callable only by the trusted application server.

create or replace function private.tournament_division_family(p_event jsonb)
returns text language sql immutable security invoker set search_path = ''
as $$
  select lower(regexp_replace(btrim(coalesce(nullif(p_event->>'event_family_label',''),
    nullif(p_event->>'label',''), 'Event')), '\s+', ' ', 'g'))
    || '|' || upper(coalesce(p_event->>'event_type',''))
    || '|' || upper(coalesce(p_event->>'gender_restriction','ANY'))
    || '|' || coalesce(p_event->>'partner_required','false')
    || '|' || upper(coalesce(p_event->>'competition_format','STANDARD'));
$$;

create or replace function private.transfer_tournament_registration_team_division(
  p_tournament_id text, p_selection_id text, p_target_event_id text,
  p_expected_updated_at timestamptz, p_context jsonb
)
returns jsonb language plpgsql security invoker set search_path = ''
as $function$
declare
  v_selection public.tournament_registration_selections%rowtype;
  v_partner public.tournament_registration_selections%rowtype;
  v_link public.tournament_registration_team_links%rowtype;
  v_source public.tournament_event_options%rowtype;
  v_target public.tournament_event_options%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_partner_registration public.tournament_registrations%rowtype;
  v_ids text[];
  v_scope text[];
  v_count integer;
  v_seats integer;
  v_rating numeric;
  v_partner_rating numeric;
  v_ceiling numeric;
  v_floor numeric;
  v_gender text;
  v_partner_gender text;
  v_age numeric;
  v_age_rule text;
  v_age_mode text;
  v_age_rules jsonb;
  v_brackets jsonb;
  v_skill_mode text;
  v_skill_label text;
  v_now timestamptz := clock_timestamp();
begin
  if p_context is null or jsonb_typeof(p_context) <> 'object' then
    return jsonb_build_object('ok', false, 'code','SELECTION_WRITE_CONFLICT','reason','PARTNER_RELATIONSHIP_CHANGED');
  end if;
  select * into v_selection from public.tournament_registration_selections
    where id = p_selection_id and tournament_id = p_tournament_id;
  if not found then
    return jsonb_build_object('ok',false,'code','SELECTION_NOT_FOUND');
  end if;
  select * into v_link from public.tournament_registration_team_links
    where id = p_context->>'team_link_id' and tournament_id::text = p_tournament_id
      and status in ('CONFIRMED','ADMIN_CONFIRMED')
      and p_selection_id in (selection1_id,selection2_id);
  if not found then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','PARTNER_RELATIONSHIP_CHANGED');
  end if;
  v_ids := array[v_link.selection1_id,v_link.selection2_id];
  -- Include all of both players' selections before taking the universal
  -- registration/selection locks: full public saves and partner edits share it.
  select array_agg(id order by id) into v_scope from public.tournament_registration_selections
    where registration_id in (v_link.registration1_id,v_link.registration2_id);
  perform private.lock_tournament_registration_selection_scope(v_scope);
  select * into v_selection from public.tournament_registration_selections where id = p_selection_id for update;
  select * into v_partner from public.tournament_registration_selections
    where id = any(v_ids) and id <> p_selection_id for update;
  select * into v_link from public.tournament_registration_team_links where id = v_link.id for update;
  select * into v_registration from public.tournament_registrations where id = v_selection.registration_id for update;
  select * into v_partner_registration from public.tournament_registrations where id = v_partner.registration_id for update;
  if v_selection.updated_at is distinct from p_expected_updated_at
     or v_link.updated_at is distinct from (p_context->>'team_link_updated_at')::timestamptz
     or v_partner.id is distinct from p_context->>'partner_selection_id'
     or v_partner.updated_at is distinct from (p_context->>'partner_selection_updated_at')::timestamptz
     or v_partner_registration.id is distinct from p_context->>'partner_registration_id'
     or v_partner_registration.updated_at is distinct from (p_context->>'partner_registration_updated_at')::timestamptz
     or v_link.status not in ('CONFIRMED','ADMIN_CONFIRMED')
     or v_selection.event_option_id is distinct from v_link.event_option_id
     or v_partner.event_option_id is distinct from v_link.event_option_id then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','STALE_TEAM_VERSION');
  end if;
  if lower(v_registration.status) in ('cancelled','withdrawn')
     or lower(v_partner_registration.status) in ('cancelled','withdrawn') then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','REGISTRATION_CANCELLED');
  end if;
  perform 1 from public.tournament_event_options
    where id in (v_selection.event_option_id,p_target_event_id) order by id for update;
  select * into v_source from public.tournament_event_options where id = v_selection.event_option_id;
  select * into v_target from public.tournament_event_options
    where id = p_target_event_id and tournament_id = p_tournament_id;
  if v_target.id is null or not coalesce(v_target.enabled,false)
     or lower(coalesce(v_target.status,'')) not in ('open','tentative','confirmed')
     or not exists (select 1 from public.tournament_registration_days
       where id = v_target.registration_day_id and tournament_id = p_tournament_id and enabled)
     or private.tournament_division_family(to_jsonb(v_source)) is distinct from private.tournament_division_family(to_jsonb(v_target))
     or coalesce(v_target.competition_format,'STANDARD') <> 'STANDARD' then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','INVALID_DIVISION_TARGET');
  end if;
  if coalesce(v_source.price_usd,0) is distinct from coalesce(v_target.price_usd,0) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','DIVISION_PRICE_CHANGED');
  end if;
  if exists (select 1 from public.tournament_commerce_orders o
      where o.registration_id in (v_registration.id,v_partner_registration.id) and o.status = 'ACTIVE')
    and (exists (select 1 from public.tournament_commerce_bundles b
      join public.tournament_commerce_bundle_components c on c.bundle_id = b.id
      where b.tournament_id::text = p_tournament_id and b.status = 'ACTIVE'
        and c.component_type in ('EVENT_OPTION','EVENT_CHOICE') and c.event_option_id in (v_source.id,v_target.id))
      or exists (select 1 from public.tournament_commerce_orders o
        join public.tournament_commerce_order_lines line on line.order_id = o.id and line.active and line.line_type = 'BUNDLE'
        cross join lateral jsonb_array_elements(coalesce(line.line_snapshot->'component_snapshot','[]'::jsonb)) component
        where o.registration_id in (v_registration.id,v_partner_registration.id) and o.status = 'ACTIVE'
          and component->>'event_option_id' in (v_source.id,v_target.id))) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','DIVISION_COMMERCE_REVIEW_REQUIRED');
  end if;
  if exists (select 1 from public.tournament_registration_partner_requests r
    where (r.requester_selection_id = any(v_ids) or r.target_selection_id = any(v_ids))
      and r.status in ('PENDING','ADMIN_CONFIRMED')
      and (r.id is distinct from v_link.accepted_request_id or r.status = 'PENDING'))
    or exists (select 1 from public.tournament_registration_team_links l
      where l.id <> v_link.id and l.status in ('CONFIRMED','ADMIN_CONFIRMED')
        and (l.selection1_id = any(v_ids) or l.selection2_id = any(v_ids))) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','PARTNER_RELATIONSHIP_CHANGED');
  end if;
  if exists (select 1 from public.tournament_registration_selections s
    join public.tournament_event_options e on e.id = s.event_option_id
    where s.registration_id in (v_registration.id,v_partner_registration.id)
      and not (s.id = any(v_ids))
      and private.tournament_division_family(to_jsonb(e)) = private.tournament_division_family(to_jsonb(v_target))) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','DUPLICATE_EVENT_FAMILY');
  end if;
  -- A target roster already generated or either event already playing must be
  -- reconciled explicitly by a director; never quietly strand a draw team.
  if exists (select 1 from public.tournament_teams t
    left join public.tournament_event_draws d on d.id = t.draw_id and d.tournament_id = t.tournament_id
    where t.tournament_id::text = p_tournament_id
      and coalesce(d.event_option_id::text,t.event_option_id::text) in (v_source.id,v_target.id)
      and ((coalesce(d.event_option_id::text,t.event_option_id::text) = v_target.id
          and upper(coalesce(t.source,'')) in ('REGISTRATION','REGISTRATION_COMBINED_RATING'))
        or t.source_selection_id::text = any(v_ids)
        or (upper(coalesce(t.source,'')) in ('REGISTRATION','REGISTRATION_COMBINED_RATING')
          and (v_registration.player_id in (t.player1_id,t.player2_id)
            or v_partner_registration.player_id in (t.player1_id,t.player2_id)))))
    or exists (select 1 from public.tournament_event_draws d
      where d.tournament_id::text = p_tournament_id and d.event_option_id::text in (v_source.id,v_target.id)
        and lower(coalesce(d.status,'')) not in ('draft','ready'))
    or exists (select 1 from public.tournament_games g
      join public.tournament_event_draws d on d.id = g.draw_id
      where d.tournament_id::text = p_tournament_id and d.event_option_id::text in (v_source.id,v_target.id)) then
    return jsonb_build_object('ok',false,'code','REGISTRATION_IMPORTED_TO_DRAW');
  end if;
  select coalesce(sum(case when upper(s.partner_mode) = 'HAS_PARTNER' and not exists (
      select 1 from public.tournament_registration_team_links l
      where l.status in ('CONFIRMED','ADMIN_CONFIRMED') and s.id in (l.selection1_id,l.selection2_id)
    ) then 2 else 1 end),0) into v_seats
    from public.tournament_registration_selections s
    join public.tournament_registrations r on r.id = s.registration_id
    where s.event_option_id = v_target.id and lower(r.status) not in ('cancelled','withdrawn')
      and not (s.id = any(v_ids));
  if v_target.capacity_teams is not null and v_target.capacity_teams > 0
     and v_seats + 2 > v_target.capacity_teams * 2 then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','DIVISION_FULL');
  end if;
  -- Recheck canonical eligibility inside the lock, in addition to the richer
  -- application diagnostics. Linked-player ratings override registration text.
  select coalesce(case when p.rating > 10 then p.rating / 400.0 else p.rating end,
    nullif(to_jsonb(p)->>'doubles_skill','')::numeric,v_registration.doubles_skill)
    into v_rating from (select 1) dummy left join public.players p on p.id = v_registration.player_id;
  select coalesce(case when p.rating > 10 then p.rating / 400.0 else p.rating end,
    nullif(to_jsonb(p)->>'doubles_skill','')::numeric,v_partner_registration.doubles_skill)
    into v_partner_rating from (select 1) dummy left join public.players p on p.id = v_partner_registration.player_id;
  select coalesce(nullif(to_jsonb(p)->>'age','')::integer,v_registration.age),
    coalesce(nullif(btrim(to_jsonb(p)->>'gender'),''),v_registration.gender)
    into v_registration.age,v_registration.gender
    from (select 1) dummy left join public.players p on p.id = v_registration.player_id;
  select coalesce(nullif(to_jsonb(p)->>'age','')::integer,v_partner_registration.age),
    coalesce(nullif(btrim(to_jsonb(p)->>'gender'),''),v_partner_registration.gender)
    into v_partner_registration.age,v_partner_registration.gender
    from (select 1) dummy left join public.players p on p.id = v_partner_registration.player_id;
  v_skill_label := regexp_replace(btrim(coalesce(v_target.skill_label,'')), '^skill\s*', '', 'i');
  v_skill_mode := case
    when v_target.eligibility_mode = 'COMBINED_RATING_CAP' or upper(v_target.skill_mode) = 'COMBINED_RATING_CAP' then 'COMBINED_RATING_CAP'
    when v_target.eligibility_mode = 'CUSTOM' or upper(v_target.skill_mode) = 'CUSTOM' then 'CUSTOM'
    when v_target.eligibility_mode = 'MINIMUM' or upper(v_target.skill_mode) in ('MINIMUM','MIN','AT_LEAST') then 'MINIMUM'
    when v_target.eligibility_mode in ('OPEN','NONE') then 'OPEN'
    when upper(v_target.skill_mode) in ('STANDARD','SKILL_BRACKET','CEILING','MAXIMUM') then 'STANDARD'
    when lower(v_skill_label) = 'open' or v_skill_label ~ '\+$' then 'OPEN'
    when v_target.eligibility_mode = 'STANDARD' or v_skill_label ~ '^[0-9]+(\.[0-9]+)?$' then 'STANDARD'
    else 'OPEN' end;
  v_floor := case when v_skill_mode in ('MINIMUM','CUSTOM') then coalesce(v_target.skill_min_rating,
    case when v_skill_mode = 'MINIMUM' and v_skill_label ~ '^[0-9]+(\.[0-9]+)?\s*\+?$'
      then regexp_replace(v_skill_label,'\s*\+$','','g')::numeric end) end;
  v_ceiling := case when v_skill_mode = 'CUSTOM' then v_target.skill_max_rating
    when v_skill_mode = 'STANDARD' then coalesce(v_target.skill_max_rating,
      case when v_skill_label ~ '^[0-9]+(\.[0-9]+)?$' then v_skill_label::numeric + 0.5 end) end;
  if (v_skill_mode = 'STANDARD' and v_ceiling is null)
    or (v_skill_mode = 'MINIMUM' and v_floor is null)
    or (v_skill_mode = 'CUSTOM' and v_ceiling is null and v_floor is null)
    or (v_floor is not null and v_rating is not null and v_partner_rating is not null and greatest(v_rating,v_partner_rating) < v_floor)
    or (v_ceiling is not null and greatest(v_rating,v_partner_rating) >= v_ceiling)
    or (v_skill_mode = 'COMBINED_RATING_CAP' and
      (greatest(v_rating,v_partner_rating) >= v_target.combined_rating_cap or round(v_rating + v_partner_rating,2) >= v_target.combined_rating_cap)) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','TEAM_NOT_ELIGIBLE');
  end if;
  v_gender := case when lower(v_registration.gender) in ('men','man','male','m') then 'MEN'
    when lower(v_registration.gender) in ('women','woman','female','f') then 'WOMEN' end;
  v_partner_gender := case when lower(v_partner_registration.gender) in ('men','man','male','m') then 'MEN'
    when lower(v_partner_registration.gender) in ('women','woman','female','f') then 'WOMEN' end;
  -- Gender mismatches continue through the existing director review/import
  -- approval flow. Missing gender data still needs completing before saving.
  if upper(v_target.gender_restriction) <> 'ANY' and
      (nullif(btrim(v_registration.gender),'') is null or nullif(btrim(v_partner_registration.gender),'') is null) then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','TEAM_NOT_ELIGIBLE');
  end if;
  -- Legacy event rows store age_rules as text; accept blank legacy values and
  -- parse the same JSON object used by registration validation.
  v_age_rules := coalesce(nullif(btrim(v_target.age_rules::text),'')::jsonb,'{}'::jsonb);
  v_age_mode := upper(coalesce(nullif(v_target.age_mode,''),v_age_rules->>'mode','ALL_AGES'));
  v_age_rule := upper(coalesce(v_age_rules->>'team_age_rule','YOUNGER'));
  if v_age_mode not in ('ALL_AGES','ALL','OPEN','ALL AGES') then
    if v_registration.age is null or v_partner_registration.age is null then
      return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','TEAM_NOT_ELIGIBLE');
    end if;
    v_age := case v_age_rule when 'OLDER' then greatest(v_registration.age,v_partner_registration.age)
      when 'AVERAGE' then (v_registration.age+v_partner_registration.age)::numeric/2
      else least(v_registration.age,v_partner_registration.age) end;
    if v_age_mode = 'SPLIT_AGE' then
      if not (least(v_registration.age,v_partner_registration.age) < (v_age_rules->>'split_age_threshold')::numeric
        and greatest(v_registration.age,v_partner_registration.age) >= (v_age_rules->>'split_age_threshold')::numeric) then
        return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','TEAM_NOT_ELIGIBLE');
      end if;
    else
      v_brackets := case when v_age_mode = 'FIXED_AGE_BRACKET' and not (v_age_rules ? 'brackets')
        then jsonb_build_array(v_age_rules) else coalesce(v_age_rules->'brackets','[]'::jsonb) end;
      if not exists (select 1 from jsonb_array_elements(v_brackets) b
        where ((b->>'min_age') is null or v_age >= (b->>'min_age')::numeric)
          and (v_age_rule <> 'BOTH_QUALIFY' or (
            ((b->>'min_age') is null or greatest(v_registration.age,v_partner_registration.age) >= (b->>'min_age')::numeric)))) then
        return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','TEAM_NOT_ELIGIBLE');
      end if;
    end if;
  end if;
  perform set_config('jupr.selection_edit_rpc','on',true);
  perform set_config('jupr.team_division_transfer',v_link.id,true);
  update public.tournament_registration_selections
    set event_option_id = v_target.id, registration_day_id = v_target.registration_day_id
    where id = any(v_ids);
  update public.tournament_registration_partner_requests
    set event_option_id = v_target.id, updated_at = greatest(v_now,updated_at + interval '1 microsecond')
    where id = v_link.accepted_request_id;
  update public.tournament_registration_team_links
    set event_option_id = v_target.id, updated_at = greatest(v_now,updated_at + interval '1 microsecond')
    where id = v_link.id;
  update public.tournament_registration_team_members set event_option_id = v_target.id
    where team_link_id = v_link.id;
  update public.tournament_registrations
    set updated_at = greatest(v_now,updated_at + interval '1 microsecond')
    where id in (v_registration.id,v_partner_registration.id);
  perform set_config('jupr.team_division_transfer','',true);
  perform public.refresh_initial_combined_rating_review_v1(v_selection.id,'division-transfer');
  perform public.refresh_initial_combined_rating_review_v1(v_partner.id,'division-transfer');
  select * into v_selection from public.tournament_registration_selections where id = p_selection_id;
  return jsonb_build_object('ok',true,'selection',to_jsonb(v_selection),
    'team_transfer',jsonb_build_object('team_link_id',v_link.id,'selection_ids',to_jsonb(v_ids),
      'previous_event_option_id',v_source.id,'event_option_id',v_target.id));
end
$function$;

revoke all on function private.tournament_division_family(jsonb) from public, anon, authenticated;
grant execute on function private.tournament_division_family(jsonb) to service_role;
revoke all on function private.transfer_tournament_registration_team_division(text,text,text,timestamptz,jsonb) from public, anon, authenticated;
grant execute on function private.transfer_tournament_registration_team_division(text,text,text,timestamptz,jsonb) to service_role;

-- Preserve the complete preceding editor contracts, including exact imported
-- roster locks. The wrappers add one atomic operation around these baselines.

create or replace function private.admin_selection_before_division_transfer(
  p_tournament_id text,
  p_selection_id text,
  p_expected_updated_at timestamptz,
  p_patch jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_before public.tournament_registration_selections%rowtype;
  v_candidate public.tournament_registration_selections%rowtype;
  v_after public.tournament_registration_selections%rowtype;
  v_target_event public.tournament_event_options%rowtype;
  v_target_day public.tournament_registration_days%rowtype;
  v_bad_key text;
  v_target_family text;
  v_event_changed boolean;
  v_relationship_sensitive boolean;
  v_updated_at timestamptz;
begin
  if nullif(btrim(p_tournament_id), '') is null
     or nullif(btrim(p_selection_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: tournament and selection IDs are required.';
  end if;

  if p_patch is null or jsonb_typeof(p_patch) <> 'object' then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: patch must be a JSON object.';
  end if;

  select patch_key
  into v_bad_key
  from jsonb_object_keys(p_patch) as patch_keys(patch_key)
  where patch_key not in (
    'registration_day_id',
    'event_option_id',
    'partner_mode',
    'partner_name',
    'partner_email',
    'partner_phone',
    'partner_dupr_id',
    'partner_skill',
    'partner_age',
    'partner_gender',
    'partner_note',
    'show_on_partner_board'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: unsupported field ' || v_bad_key || '.';
  end if;

  if not exists (
    select 1
    from public.tournament_registration_selections as selection
    where selection.id = btrim(p_selection_id)
      and selection.tournament_id = btrim(p_tournament_id)
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_NOT_FOUND'
    );
  end if;

  begin
    perform private.lock_tournament_registration_selection_scope(
      array[btrim(p_selection_id)]
    );
  exception
    when others then
      if sqlerrm like 'JUPR_RELATION_SELECTION_NOT_FOUND:%' then
        return jsonb_build_object(
          'ok', false,
          'code', 'SELECTION_NOT_FOUND'
        );
      end if;
      raise;
  end;

  select selection.*
  into v_before
  from public.tournament_registration_selections as selection
  where selection.id = btrim(p_selection_id)
    and selection.tournament_id = btrim(p_tournament_id)
  for update;

  if not found then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_NOT_FOUND'
    );
  end if;

  if p_expected_updated_at is null
     or v_before.updated_at is distinct from p_expected_updated_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  select populated.*
  into v_candidate
  from jsonb_populate_record(v_before, p_patch) as populated;

  v_candidate.event_option_id := nullif(btrim(v_candidate.event_option_id), '');
  v_candidate.partner_mode := upper(nullif(btrim(v_candidate.partner_mode), ''));

  if v_candidate.event_option_id is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: event option is required.';
  end if;

  if v_candidate.partner_mode is null
     or v_candidate.partner_mode not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER') then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: invalid partner mode.';
  end if;

  select event_option.*
  into v_target_event
  from public.tournament_event_options as event_option
  where event_option.id = v_candidate.event_option_id
    and event_option.tournament_id = btrim(p_tournament_id)
  for share;

  if not found then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: event option is not in this tournament.';
  end if;

  if p_patch ? 'registration_day_id'
     and nullif(btrim(p_patch ->> 'registration_day_id'), '')
       is distinct from v_target_event.registration_day_id then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_TARGET: registration day does not match the event option.';
  end if;

  v_candidate.registration_day_id := v_target_event.registration_day_id;
  v_event_changed :=
    v_candidate.event_option_id is distinct from v_before.event_option_id;

  if v_event_changed then
    select registration_day.*
    into v_target_day
    from public.tournament_registration_days as registration_day
    where registration_day.id = v_target_event.registration_day_id
      and registration_day.tournament_id = btrim(p_tournament_id)
    for share;

    if not found
       or not v_target_day.enabled
       or not v_target_event.enabled
       or lower(coalesce(nullif(btrim(v_target_event.status), ''), 'draft'))
         not in ('open', 'tentative', 'confirmed') then
      raise exception using
        errcode = '22023',
        message = 'JUPR_SELECTION_INVALID_TARGET: target event is not open on an enabled registration day.';
    end if;
  end if;

  v_target_family := lower(
    regexp_replace(
      btrim(
        coalesce(
          nullif(v_target_event.event_family_label, ''),
          nullif(v_target_event.label, ''),
          'Event'
        )
      ),
      '\s+',
      ' ',
      'g'
    )
  );

  if exists (
    select 1
    from public.tournament_registration_selections as sibling
    join public.tournament_event_options as sibling_event
      on sibling_event.id = sibling.event_option_id
    where sibling.registration_id = v_before.registration_id
      and sibling.id <> v_before.id
      and sibling.registration_day_id = v_target_event.registration_day_id
      and lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(sibling_event.event_family_label, ''),
              nullif(sibling_event.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      ) = v_target_family
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'DUPLICATE_EVENT_FAMILY'
    );
  end if;

  v_relationship_sensitive :=
    v_candidate.event_option_id is distinct from v_before.event_option_id
    or v_candidate.registration_day_id is distinct from v_before.registration_day_id
    or v_candidate.partner_mode is distinct from v_before.partner_mode
    or v_candidate.partner_name is distinct from v_before.partner_name
    or v_candidate.partner_email is distinct from v_before.partner_email
    or v_candidate.partner_phone is distinct from v_before.partner_phone
    or v_candidate.partner_dupr_id is distinct from v_before.partner_dupr_id
    or v_candidate.partner_skill is distinct from v_before.partner_skill
    or v_candidate.partner_age is distinct from v_before.partner_age
    or v_candidate.partner_gender is distinct from v_before.partner_gender;

  if v_relationship_sensitive and (
    exists (
      select 1
      from public.tournament_registration_partner_requests as request
      where request.status in ('PENDING', 'ADMIN_CONFIRMED')
        and (
          request.requester_selection_id = v_before.id
          or request.target_selection_id = v_before.id
        )
    )
    or exists (
      select 1
      from public.tournament_registration_team_links as team_link
      where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          team_link.selection1_id = v_before.id
          or team_link.selection2_id = v_before.id
        )
    )
    or exists (
      select 1
      from public.tournament_registration_team_members as team_member
      where team_member.status = 'ACTIVE'
        and team_member.selection_id = v_before.id
    )
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'PARTNER_RELATIONSHIP_CHANGED'
    );
  end if;

  if v_candidate.partner_mode = 'HAS_PARTNER'
     and not exists (
       select 1
       from public.tournament_registration_team_links as team_link
       where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
         and (
           team_link.selection1_id = v_before.id
           or team_link.selection2_id = v_before.id
         )
     )
     and (
       nullif(btrim(v_candidate.partner_name), '') is null
       or nullif(btrim(v_candidate.partner_email), '') is null
       or v_candidate.partner_age is null
       or nullif(btrim(v_candidate.partner_gender), '') is null
     ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_SELECTION_INVALID_PATCH: manual partner name, email, age, and gender are required.';
  end if;

  if v_candidate.partner_mode in ('NONE', 'NEEDS_PARTNER') then
    v_candidate.partner_name := null;
    v_candidate.partner_email := null;
    v_candidate.partner_phone := null;
    v_candidate.partner_dupr_id := null;
    v_candidate.partner_skill := null;
    v_candidate.partner_age := null;
    v_candidate.partner_gender := null;
  end if;
  if v_candidate.partner_mode <> 'NEEDS_PARTNER' then
    v_candidate.show_on_partner_board := false;
  end if;

  v_updated_at := greatest(
    pg_catalog.clock_timestamp(),
    v_before.updated_at + interval '1 microsecond'
  );

  perform pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true);

  update public.tournament_registration_selections as selection
  set
    registration_day_id = v_candidate.registration_day_id,
    event_option_id = v_candidate.event_option_id,
    partner_mode = v_candidate.partner_mode,
    partner_name = v_candidate.partner_name,
    partner_email = v_candidate.partner_email,
    partner_phone = v_candidate.partner_phone,
    partner_dupr_id = v_candidate.partner_dupr_id,
    partner_skill = v_candidate.partner_skill,
    partner_age = v_candidate.partner_age,
    partner_gender = v_candidate.partner_gender,
    partner_note = v_candidate.partner_note,
    show_on_partner_board = v_candidate.show_on_partner_board,
    updated_at = v_updated_at
  where selection.id = v_before.id
    and selection.tournament_id = btrim(p_tournament_id)
    and selection.updated_at = p_expected_updated_at
  returning selection.* into v_after;

  if not found then
    return jsonb_build_object(
      'ok', false,
      'code', 'SELECTION_WRITE_CONFLICT',
      'reason', 'STALE_VERSION'
    );
  end if;

  return jsonb_build_object(
    'ok', true,
    'selection', to_jsonb(v_after)
  );
end
$function$;

create or replace function private.public_edit_before_division_transfer(
  p_tournament_id text,
  p_registration_id text,
  p_expected_updated_at timestamptz,
  p_expected_selection_versions jsonb,
  p_registration_patch jsonb,
  p_selections jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration public.tournament_registrations%rowtype;
  v_existing_ids text[];
  v_bad_key text;
  v_updated_at timestamptz;
  v_selection_count integer;
begin
  if nullif(btrim(p_tournament_id), '') is null
     or nullif(btrim(p_registration_id), '') is null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: tournament and registration IDs are required.';
  end if;

  if p_expected_updated_at is null
     or p_expected_selection_versions is null
     or jsonb_typeof(p_expected_selection_versions) <> 'array'
     or p_registration_patch is null
     or jsonb_typeof(p_registration_patch) <> 'object'
     or p_selections is null
     or jsonb_typeof(p_selections) <> 'array'
     or jsonb_array_length(p_selections) = 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: versions, registration patch, and at least one selection are required.';
  end if;

  select patch_key
  into v_bad_key
  from jsonb_object_keys(p_registration_patch) as patch_keys(patch_key)
  where patch_key not in (
    'first_name',
    'last_name',
    'display_name',
    'phone',
    'dupr_id',
    'doubles_skill',
    'singles_skill',
    'age',
    'age_bracket',
    'gender',
    'notes',
    'wants_partner_board_contact'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: unsupported registration field ' || v_bad_key || '.';
  end if;

  select selection_key
  into v_bad_key
  from jsonb_array_elements(p_selections) as desired(selection_json)
  cross join lateral jsonb_object_keys(desired.selection_json) as selection_keys(selection_key)
  where selection_key not in (
    'id',
    'tournament_id',
    'registration_id',
    'registration_day_id',
    'event_option_id',
    'partner_mode',
    'partner_name',
    'partner_email',
    'partner_phone',
    'partner_dupr_id',
    'partner_skill',
    'partner_age',
    'partner_gender',
    'partner_note',
    'show_on_partner_board',
    'sort_order',
    'created_at',
    'updated_at'
  )
  limit 1;

  if v_bad_key is not null then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: unsupported selection field ' || v_bad_key || '.';
  end if;

  select array_agg(selection.id order by selection.id)
  into v_existing_ids
  from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  if coalesce(cardinality(v_existing_ids), 0) > 0 then
    perform private.lock_tournament_registration_selection_scope(v_existing_ids);
  else
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-registration:' || btrim(p_registration_id),
        0
      )
    );
  end if;

  select registration.*
  into v_registration
  from public.tournament_registrations as registration
  where registration.tournament_id = btrim(p_tournament_id)
    and registration.id = btrim(p_registration_id)
  for update;

  if not found then
    return jsonb_build_object('ok', false, 'code', 'REGISTRATION_NOT_FOUND');
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as selection
    join public.tournament_teams as team
      on team.tournament_id::text = selection.tournament_id
     and upper(coalesce(team.source, '')) in (
       'REGISTRATION',
       'REGISTRATION_COMBINED_RATING'
     )
     and (
       team.source_selection_id::text = selection.id
       or (
         v_registration.player_id is not null
         and v_registration.player_id in (team.player1_id, team.player2_id)
       )
     )
    left join public.tournament_event_draws as draw
      on draw.id = team.draw_id
     and draw.tournament_id = team.tournament_id
    where selection.tournament_id = btrim(p_tournament_id)
      and selection.registration_id = btrim(p_registration_id)
      and coalesce(
        draw.registration_day_id::text,
        team.registration_day_id::text
      ) = selection.registration_day_id
      and coalesce(
        draw.event_option_id::text,
        team.event_option_id::text
      ) = selection.event_option_id
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_IMPORTED_TO_DRAW'
    );
  end if;

  if v_registration.updated_at is distinct from p_expected_updated_at then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_EDIT_CONFLICT',
      'reason', 'STALE_REGISTRATION_VERSION'
    );
  end if;

  if jsonb_array_length(p_expected_selection_versions) <> (
    select count(distinct nullif(btrim(expected.id), ''))
    from jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
  ) or exists (
    select 1
    from public.tournament_registration_selections as selection
    left join jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
      on expected.id = selection.id
     and expected.updated_at = selection.updated_at
    where selection.tournament_id = btrim(p_tournament_id)
      and selection.registration_id = btrim(p_registration_id)
      and expected.id is null
  ) or exists (
    select 1
    from jsonb_to_recordset(p_expected_selection_versions)
      as expected(id text, updated_at timestamptz)
    left join public.tournament_registration_selections as selection
      on selection.id = expected.id
     and selection.tournament_id = btrim(p_tournament_id)
     and selection.registration_id = btrim(p_registration_id)
     and selection.updated_at = expected.updated_at
    where selection.id is null
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_EDIT_CONFLICT',
      'reason', 'STALE_SELECTION_VERSION'
    );
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    )
    where nullif(btrim(desired.id), '') is null
       or nullif(btrim(desired.registration_day_id), '') is null
       or nullif(btrim(desired.event_option_id), '') is null
       or upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE'))
         not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER')
  ) or (
    select count(*)
    from jsonb_to_recordset(p_selections) as desired(id text)
  ) <> (
    select count(distinct btrim(desired.id))
    from jsonb_to_recordset(p_selections) as desired(id text)
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: selection identities are missing, duplicated, or invalid.';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text
    )
    left join public.tournament_registration_selections as conflicting
      on conflicting.id = desired.id
    left join public.tournament_event_options as event_option
      on event_option.id = desired.event_option_id
     and event_option.tournament_id = btrim(p_tournament_id)
    left join public.tournament_registration_days as registration_day
      on registration_day.id = desired.registration_day_id
     and registration_day.tournament_id = btrim(p_tournament_id)
    left join public.tournament_registration_selections as existing
      on existing.id = desired.id
     and existing.tournament_id = btrim(p_tournament_id)
     and existing.registration_id = btrim(p_registration_id)
    where event_option.id is null
       or registration_day.id is null
       or event_option.registration_day_id <> desired.registration_day_id
       or (
         conflicting.id is not null
         and (
           conflicting.tournament_id <> btrim(p_tournament_id)
           or conflicting.registration_id <> btrim(p_registration_id)
         )
       )
       or (
         (existing.id is null or existing.event_option_id <> desired.event_option_id)
         and (
           not coalesce(event_option.enabled, false)
           or not coalesce(registration_day.enabled, false)
           or lower(coalesce(nullif(btrim(event_option.status), ''), 'draft'))
             not in ('open', 'tentative', 'confirmed')
         )
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: a selection target is unavailable or belongs to another registration.';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text
    )
    join public.tournament_event_options as event_option
      on event_option.id = desired.event_option_id
    group by
      desired.registration_day_id,
      lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(event_option.event_family_label, ''),
              nullif(event_option.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      )
    having count(*) > 1
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_REGISTRATION_EDIT_INVALID: choose one division per event family and day.';
  end if;

  if exists (
    select 1
    from public.tournament_registration_selections as existing
    left join jsonb_to_recordset(p_selections) as desired(
      id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    ) on desired.id = existing.id
    where existing.tournament_id = btrim(p_tournament_id)
      and existing.registration_id = btrim(p_registration_id)
      and (
        (
          desired.id is null
          and (
            exists (
              select 1
              from public.tournament_registration_partner_requests as request
              where request.requester_selection_id = existing.id
                 or request.target_selection_id = existing.id
            )
            or exists (
              select 1
              from public.tournament_registration_team_links as team_link
              where team_link.selection1_id = existing.id
                 or team_link.selection2_id = existing.id
            )
            or exists (
              select 1
              from public.tournament_registration_team_members as team_member
              where team_member.selection_id = existing.id
            )
          )
        )
        or (
          desired.id is not null
          and (
            desired.registration_day_id is distinct from existing.registration_day_id
            or desired.event_option_id is distinct from existing.event_option_id
            or upper(coalesce(desired.partner_mode, 'NONE'))
              is distinct from upper(coalesce(existing.partner_mode, 'NONE'))
          )
          and (
            exists (
              select 1
              from public.tournament_registration_partner_requests as request
              where request.status in ('PENDING', 'ADMIN_CONFIRMED')
                and (
                  request.requester_selection_id = existing.id
                  or request.target_selection_id = existing.id
                )
            )
            or exists (
              select 1
              from public.tournament_registration_team_links as team_link
              where team_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
                and (
                  team_link.selection1_id = existing.id
                  or team_link.selection2_id = existing.id
                )
            )
            or exists (
              select 1
              from public.tournament_registration_team_members as team_member
              where team_member.status = 'ACTIVE'
                and team_member.selection_id = existing.id
            )
          )
        )
      )
  ) then
    return jsonb_build_object(
      'ok', false,
      'code', 'REGISTRATION_RELATIONSHIP_LOCKED'
    );
  end if;

  delete from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id)
    and not exists (
      select 1
      from jsonb_to_recordset(p_selections) as desired(id text)
      where desired.id = selection.id
    );

  perform pg_catalog.set_config('jupr.selection_edit_rpc', 'on', true);

  update public.tournament_registration_selections as selection
  set
    registration_day_id = desired.registration_day_id,
    event_option_id = desired.event_option_id,
    partner_mode = upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE')),
    partner_name = nullif(btrim(desired.partner_name), ''),
    partner_email = nullif(lower(btrim(desired.partner_email)), ''),
    partner_phone = nullif(btrim(desired.partner_phone), ''),
    partner_dupr_id = nullif(btrim(desired.partner_dupr_id), ''),
    partner_skill = desired.partner_skill,
    partner_age = desired.partner_age,
    partner_gender = nullif(btrim(desired.partner_gender), ''),
    partner_note = nullif(btrim(desired.partner_note), ''),
    show_on_partner_board = coalesce(desired.show_on_partner_board, false),
    sort_order = desired.sort_order
  from jsonb_to_recordset(p_selections) as desired(
    id text,
    registration_day_id text,
    event_option_id text,
    partner_mode text,
    partner_name text,
    partner_email text,
    partner_phone text,
    partner_dupr_id text,
    partner_skill numeric,
    partner_age integer,
    partner_gender text,
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer
  )
  where selection.id = desired.id
    and selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  insert into public.tournament_registration_selections (
    id,
    tournament_id,
    registration_id,
    registration_day_id,
    event_option_id,
    partner_mode,
    partner_name,
    partner_email,
    partner_phone,
    partner_dupr_id,
    partner_skill,
    partner_age,
    partner_gender,
    partner_note,
    show_on_partner_board,
    sort_order,
    created_at,
    updated_at
  )
  select
    desired.id,
    btrim(p_tournament_id),
    btrim(p_registration_id),
    desired.registration_day_id,
    desired.event_option_id,
    upper(coalesce(nullif(btrim(desired.partner_mode), ''), 'NONE')),
    nullif(btrim(desired.partner_name), ''),
    nullif(lower(btrim(desired.partner_email)), ''),
    nullif(btrim(desired.partner_phone), ''),
    nullif(btrim(desired.partner_dupr_id), ''),
    desired.partner_skill,
    desired.partner_age,
    nullif(btrim(desired.partner_gender), ''),
    nullif(btrim(desired.partner_note), ''),
    coalesce(desired.show_on_partner_board, false),
    desired.sort_order,
    coalesce(desired.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from jsonb_to_recordset(p_selections) as desired(
    id text,
    registration_day_id text,
    event_option_id text,
    partner_mode text,
    partner_name text,
    partner_email text,
    partner_phone text,
    partner_dupr_id text,
    partner_skill numeric,
    partner_age integer,
    partner_gender text,
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer,
    created_at timestamptz
  )
  where not exists (
    select 1
    from public.tournament_registration_selections as existing
    where existing.id = desired.id
  );

  v_updated_at := greatest(
    pg_catalog.clock_timestamp(),
    v_registration.updated_at + interval '1 microsecond'
  );

  update public.tournament_registrations as registration
  set
    first_name = nullif(btrim(p_registration_patch ->> 'first_name'), ''),
    last_name = nullif(btrim(p_registration_patch ->> 'last_name'), ''),
    display_name = nullif(btrim(p_registration_patch ->> 'display_name'), ''),
    phone = nullif(btrim(p_registration_patch ->> 'phone'), ''),
    dupr_id = nullif(btrim(p_registration_patch ->> 'dupr_id'), ''),
    doubles_skill = (p_registration_patch ->> 'doubles_skill')::numeric,
    singles_skill = (p_registration_patch ->> 'singles_skill')::numeric,
    age = (p_registration_patch ->> 'age')::integer,
    age_bracket = nullif(btrim(p_registration_patch ->> 'age_bracket'), ''),
    gender = nullif(btrim(p_registration_patch ->> 'gender'), ''),
    notes = nullif(btrim(p_registration_patch ->> 'notes'), ''),
    wants_partner_board_contact = coalesce(
      (p_registration_patch ->> 'wants_partner_board_contact')::boolean,
      false
    ),
    updated_at = v_updated_at
  where registration.tournament_id = btrim(p_tournament_id)
    and registration.id = btrim(p_registration_id)
    and registration.updated_at = p_expected_updated_at;

  if not found then
    raise exception using
      errcode = '40001',
      message = 'JUPR_REGISTRATION_EDIT_CONFLICT: registration version changed during the atomic edit.';
  end if;

  select count(*)
  into v_selection_count
  from public.tournament_registration_selections as selection
  where selection.tournament_id = btrim(p_tournament_id)
    and selection.registration_id = btrim(p_registration_id);

  return jsonb_build_object(
    'ok', true,
    'registration_id', btrim(p_registration_id),
    'updated_at', v_updated_at,
    'selection_count', v_selection_count
  );
end
$function$;

create or replace function private.canonicalize_tournament_manual_partner_team_v1()
returns trigger
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_partner_name text := nullif(pg_catalog.btrim(coalesce(new.partner_name, '')), '');
  v_partner_email text := pg_catalog.lower(nullif(pg_catalog.btrim(coalesce(new.partner_email, '')), ''));
  v_partner_skill numeric := new.partner_skill;
  v_club_id text;
  v_now timestamptz := pg_catalog.clock_timestamp();
  v_primary_registration public.tournament_registrations%rowtype;
  v_partner_registration public.tournament_registrations%rowtype;
  v_player_registration public.tournament_registrations%rowtype;
  v_partner_selection public.tournament_registration_selections%rowtype;
  v_partner_player public.players%rowtype;
  v_player_match_count integer := 0;
  v_player_match_id integer;
  v_existing_link public.tournament_registration_team_links%rowtype;
  v_request_id text;
  v_link_id text;
  v_first_name text;
  v_last_name text;
  v_sort_order integer;
begin
  -- Only the transaction moving this exact existing team may suppress the
  -- temporary half-moved selection snapshot. Its relationship checks remain on.
  if exists (
    select 1 from public.tournament_registration_team_links link
    where link.id = pg_catalog.current_setting('jupr.team_division_transfer',true)
      and link.tournament_id::text = new.tournament_id
      and link.status in ('CONFIRMED','ADMIN_CONFIRMED')
      and new.id in (link.selection1_id,link.selection2_id)
  ) then
    return new;
  end if;
  if pg_catalog.upper(coalesce(new.partner_mode, 'NONE')) <> 'HAS_PARTNER' then
    return new;
  end if;

  -- A linked partner's reciprocal selection also uses HAS_PARTNER, but carries
  -- no free-text partner identity. That row must not recurse into another team.
  if v_partner_name is null and v_partner_email is null then
    return new;
  end if;
  if v_partner_name is null then
    raise exception 'JUPR_MANUAL_PARTNER_NAME_REQUIRED: enter the partner name.';
  end if;
  if v_partner_email is null then
    raise exception 'JUPR_MANUAL_PARTNER_EMAIL_REQUIRED: enter the partner email so the canonical registration can be created.';
  end if;

  select registration.*
    into v_primary_registration
    from public.tournament_registrations as registration
   where registration.id = new.registration_id
     and registration.tournament_id::text = new.tournament_id::text;
  if v_primary_registration.id is null then
    raise exception 'JUPR_MANUAL_PARTNER_PRIMARY_MISSING: primary tournament registration was not found.';
  end if;
  if pg_catalog.lower(coalesce(v_primary_registration.email, '')) = v_partner_email then
    raise exception 'JUPR_MANUAL_PARTNER_SELF: a registrant cannot be their own partner.';
  end if;

  select tournament.club_id::text
    into v_club_id
    from public.tournaments as tournament
   where tournament.id::text = new.tournament_id::text;
  if nullif(v_club_id, '') is null then
    raise exception 'JUPR_MANUAL_PARTNER_TOURNAMENT_MISSING: tournament was not found.';
  end if;

  -- Serialize manual canonicalization for the tournament. A registration can
  -- submit several divisions (and different partners) in one statement, so a
  -- single lock prevents duplicate partner identities without multi-lock
  -- ordering deadlocks across concurrent registrations.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'manual-tournament-partner:' || new.tournament_id::text,
      0
    )
  );

  select link.*
    into v_existing_link
    from public.tournament_registration_team_links as link
   where link.tournament_id::text = new.tournament_id::text
     and link.event_option_id = new.event_option_id
     and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
     and new.id in (link.selection1_id, link.selection2_id)
   limit 1;
  if v_existing_link.id is not null then
    return new;
  end if;

  select registration.*
    into v_partner_registration
    from public.tournament_registrations as registration
   where registration.tournament_id::text = new.tournament_id::text
     and pg_catalog.lower(pg_catalog.btrim(registration.email)) = v_partner_email
   limit 1;
  if v_partner_registration.id = new.registration_id then
    raise exception 'JUPR_MANUAL_PARTNER_SELF: a registrant cannot be their own partner.';
  end if;
  if v_partner_registration.id is not null
     and pg_catalog.upper(coalesce(v_partner_registration.status, 'CONFIRMED')) in ('CANCELLED', 'WITHDRAWN') then
    raise exception 'JUPR_MANUAL_PARTNER_CANCELLED: the matching partner registration is cancelled.';
  end if;

  if v_partner_registration.player_id is not null then
    select player.*
      into v_partner_player
      from public.players as player
     where player.id = v_partner_registration.player_id
       and player.club_id::text = v_club_id;
    if v_partner_player.id is null then
      raise exception 'JUPR_MANUAL_PARTNER_CLUB_MISMATCH: the matching registration links to another club.';
    end if;
  else
    select pg_catalog.count(*)::integer, pg_catalog.min(player.id)
      into v_player_match_count, v_player_match_id
      from public.players as player
     where player.club_id::text = v_club_id
       and pg_catalog.lower(pg_catalog.btrim(player.name)) = pg_catalog.lower(v_partner_name);
    if v_player_match_count > 1 then
      raise exception 'JUPR_MANUAL_PARTNER_AMBIGUOUS: more than one club player has this exact name; link the player explicitly before saving.';
    end if;
    if v_player_match_count = 1 then
      select player.*
        into v_partner_player
        from public.players as player
       where player.id = v_player_match_id;
    end if;
  end if;

  if v_partner_player.id is null then
    if v_partner_skill is null or v_partner_skill < 1 or v_partner_skill > 7 then
      raise exception 'JUPR_MANUAL_PARTNER_RATING_REQUIRED: enter a starting JUPR from 1.0 through 7.0 for the new partner.';
    end if;
    insert into public.players (
      club_id,
      name,
      rating,
      starting_rating,
      wins,
      losses,
      matches_played,
      active,
      last_game_at,
      inactive_at
    ) values (
      v_club_id,
      v_partner_name,
      v_partner_skill * 400.0,
      v_partner_skill * 400.0,
      0,
      0,
      0,
      true,
      null,
      null
    )
    returning * into v_partner_player;
  end if;

  -- Email resolution may find an older unlinked registration while the matched
  -- club player already has the canonical tournament registration. Prefer that
  -- player-linked row instead of violating the tournament/player unique index.
  if v_partner_registration.player_id is null then
    select registration.*
      into v_player_registration
      from public.tournament_registrations as registration
     where registration.tournament_id::text = new.tournament_id::text
       and registration.player_id = v_partner_player.id
     limit 1;
    if v_player_registration.id is not null then
      v_partner_registration := v_player_registration;
    end if;
  end if;

  if v_partner_registration.id is null then
    select registration.*
      into v_partner_registration
      from public.tournament_registrations as registration
     where registration.tournament_id::text = new.tournament_id::text
       and registration.player_id = v_partner_player.id
     limit 1;
  end if;

  if v_partner_registration.id is null then
    v_first_name := pg_catalog.split_part(v_partner_name, ' ', 1);
    v_last_name := nullif(
      pg_catalog.btrim(pg_catalog.substr(v_partner_name, pg_catalog.length(v_first_name) + 1)),
      ''
    );
    insert into public.tournament_registrations (
      id,
      tournament_id,
      submitted_at,
      updated_at,
      status,
      payment_status,
      first_name,
      last_name,
      display_name,
      email,
      phone,
      player_id,
      dupr_id,
      doubles_skill,
      singles_skill,
      age,
      gender,
      notes,
      wants_partner_board_contact
    ) values (
      'reg_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
      new.tournament_id,
      v_now,
      v_now,
      'confirmed',
      'unpaid',
      v_first_name,
      v_last_name,
      v_partner_name,
      v_partner_email,
      nullif(pg_catalog.btrim(coalesce(new.partner_phone, '')), ''),
      v_partner_player.id,
      nullif(pg_catalog.btrim(coalesce(new.partner_dupr_id, '')), ''),
      v_partner_skill,
      v_partner_skill,
      new.partner_age,
      nullif(pg_catalog.btrim(coalesce(new.partner_gender, '')), ''),
      'Created from a submitted tournament partner.',
      false
    )
    returning * into v_partner_registration;
  elsif v_partner_registration.player_id is null then
    update public.tournament_registrations
       set player_id = v_partner_player.id,
           updated_at = v_now
     where id = v_partner_registration.id
    returning * into v_partner_registration;
  end if;

  if v_partner_registration.id = new.registration_id then
    raise exception 'JUPR_MANUAL_PARTNER_SELF: a registrant cannot be their own partner.';
  end if;

  select selection.*
    into v_partner_selection
    from public.tournament_registration_selections as selection
   where selection.tournament_id::text = new.tournament_id::text
     and selection.registration_id = v_partner_registration.id
     and selection.event_option_id = new.event_option_id
   limit 1;

  if v_partner_selection.id is null then
    select coalesce(pg_catalog.max(selection.sort_order), -1) + 1
      into v_sort_order
      from public.tournament_registration_selections as selection
     where selection.tournament_id::text = new.tournament_id::text
       and selection.registration_id = v_partner_registration.id;
    insert into public.tournament_registration_selections (
      id,
      tournament_id,
      registration_id,
      registration_day_id,
      event_option_id,
      partner_mode,
      show_on_partner_board,
      sort_order,
      created_at,
      updated_at
    ) values (
      'sel_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
      new.tournament_id,
      v_partner_registration.id,
      new.registration_day_id,
      new.event_option_id,
      'HAS_PARTNER',
      false,
      v_sort_order,
      v_now,
      v_now
    )
    returning * into v_partner_selection;
  end if;

  if exists (
    select 1
      from public.tournament_registration_team_members as member
      join public.tournament_registration_team_links as link
        on link.id = member.team_link_id
     where member.event_option_id = new.event_option_id
       and member.selection_id in (new.id, v_partner_selection.id)
       and member.status = 'ACTIVE'
       and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
  ) then
    raise exception 'JUPR_MANUAL_PARTNER_ALREADY_LINKED: one of these entries already belongs to another confirmed team.';
  end if;

  v_request_id := 'preq_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', '');
  v_link_id := 'tlink_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', '');
  insert into public.tournament_registration_partner_requests (
    id,
    tournament_id,
    event_option_id,
    requester_selection_id,
    requester_registration_id,
    requester_player_id,
    target_selection_id,
    target_registration_id,
    target_player_id,
    target_display_name_snapshot,
    status,
    source,
    created_at,
    updated_at,
    responded_at,
    created_by_registration_id,
    created_by_user_id
  ) values (
    v_request_id,
    new.tournament_id::uuid,
    new.event_option_id,
    new.id,
    new.registration_id,
    v_primary_registration.player_id,
    v_partner_selection.id,
    v_partner_registration.id,
    v_partner_player.id,
    v_partner_registration.display_name,
    'ADMIN_CONFIRMED',
    'ADMIN_CREATED',
    v_now,
    v_now,
    v_now,
    new.registration_id,
    'manual-partner-canonicalizer'
  );

  insert into public.tournament_registration_team_links (
    id,
    tournament_id,
    event_option_id,
    registration1_id,
    registration2_id,
    selection1_id,
    selection2_id,
    player1_id,
    player2_id,
    status,
    accepted_request_id,
    created_at,
    updated_at,
    created_by_user_id
  ) values (
    v_link_id,
    new.tournament_id::uuid,
    new.event_option_id,
    new.registration_id,
    v_partner_registration.id,
    new.id,
    v_partner_selection.id,
    v_primary_registration.player_id,
    v_partner_player.id,
    'ADMIN_CONFIRMED',
    v_request_id,
    v_now,
    v_now,
    'manual-partner-canonicalizer'
  );

  insert into public.tournament_registration_team_members (
    id,
    team_link_id,
    tournament_id,
    event_option_id,
    selection_id,
    registration_id,
    player_id,
    player_order,
    status,
    created_at
  ) values
    (
      'tmem_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
      v_link_id,
      new.tournament_id::uuid,
      new.event_option_id,
      new.id,
      new.registration_id,
      v_primary_registration.player_id,
      1,
      'ACTIVE',
      v_now
    ),
    (
      'tmem_' || pg_catalog.replace(pg_catalog.gen_random_uuid()::text, '-', ''),
      v_link_id,
      new.tournament_id::uuid,
      new.event_option_id,
      v_partner_selection.id,
      v_partner_registration.id,
      v_partner_player.id,
      2,
      'ACTIVE',
      v_now
    );

  update public.tournament_registration_selections
     set partner_mode = 'HAS_PARTNER',
         show_on_partner_board = false,
         updated_at = v_now
   where id in (new.id, v_partner_selection.id)
     and (
       partner_mode is distinct from 'HAS_PARTNER'
       or show_on_partner_board is distinct from false
     );

  -- The partner selection's ordinary AFTER INSERT review runs before this
  -- trigger has created the canonical team link and members. Refresh both
  -- sides now that the complete team exists so combined-rating eligibility is
  -- based on the same canonical pair shown on the public roster.
  perform public.refresh_initial_combined_rating_review_v1(
    new.id::text,
    'manual-partner-canonicalizer'
  );
  perform public.refresh_initial_combined_rating_review_v1(
    v_partner_selection.id::text,
    'manual-partner-canonicalizer'
  );

  update public.tournament_registration_partner_requests
     set status = 'CANCELLED',
         updated_at = v_now,
         responded_at = coalesce(responded_at, v_now)
   where event_option_id = new.event_option_id
     and status = 'PENDING'
     and id <> v_request_id
     and (
       requester_selection_id in (new.id, v_partner_selection.id)
       or target_selection_id in (new.id, v_partner_selection.id)
     );

  return new;
end
$function$;

revoke all on function private.admin_selection_before_division_transfer(text,text,timestamptz,jsonb) from public, anon, authenticated;
grant execute on function private.admin_selection_before_division_transfer(text,text,timestamptz,jsonb) to service_role;
revoke all on function private.public_edit_before_division_transfer(text,text,timestamptz,jsonb,jsonb,jsonb) from public, anon, authenticated;
grant execute on function private.public_edit_before_division_transfer(text,text,timestamptz,jsonb,jsonb,jsonb) to service_role;
revoke all on function private.canonicalize_tournament_manual_partner_team_v1() from public, anon, authenticated;

create or replace function public.admin_update_tournament_registration_selection(
  p_tournament_id text, p_selection_id text, p_expected_updated_at timestamptz, p_patch jsonb
)
returns jsonb language plpgsql security invoker set search_path = ''
as $function$
declare
  v_before public.tournament_registration_selections%rowtype;
  v_result jsonb;
  v_transfer jsonb;
  v_patch jsonb;
  v_key text;
begin
  select * into v_before from public.tournament_registration_selections
    where id = p_selection_id and tournament_id = p_tournament_id;
  if v_before.id is null or not (p_patch ? 'event_option_id')
     or p_patch->>'event_option_id' is not distinct from v_before.event_option_id
     or not exists (select 1 from public.tournament_registration_team_links
       where status in ('CONFIRMED','ADMIN_CONFIRMED') and p_selection_id in (selection1_id,selection2_id)) then
    return private.admin_selection_before_division_transfer(
      p_tournament_id,p_selection_id,p_expected_updated_at,p_patch - 'division_transfer_context');
  end if;
  for v_key in select jsonb_object_keys(p_patch) loop
    if v_key not in ('event_option_id','registration_day_id','division_transfer_context','partner_note','show_on_partner_board')
       and p_patch->v_key is distinct from to_jsonb(v_before)->v_key then
      return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','PARTNER_RELATIONSHIP_CHANGED');
    end if;
  end loop;
  if p_patch ? 'registration_day_id' and not exists (select 1 from public.tournament_event_options
    where id = p_patch->>'event_option_id' and registration_day_id = p_patch->>'registration_day_id') then
    return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','INVALID_DIVISION_TARGET');
  end if;
  begin
    v_transfer := private.transfer_tournament_registration_team_division(p_tournament_id,p_selection_id,
      p_patch->>'event_option_id',p_expected_updated_at,p_patch->'division_transfer_context');
    if (v_transfer->>'ok') is distinct from 'true' then
      return v_transfer;
    end if;
    v_patch := p_patch - 'division_transfer_context' - 'event_option_id' - 'registration_day_id';
    v_result := private.admin_selection_before_division_transfer(p_tournament_id,p_selection_id,
      (v_transfer->'selection'->>'updated_at')::timestamptz,v_patch);
    if (v_result->>'ok') is distinct from 'true' then
      raise exception using errcode='PT409',message=v_result::text;
    end if;
    return v_result || jsonb_build_object('team_transfer',v_transfer->'team_transfer');
  exception when sqlstate 'PT409' then
    return sqlerrm::jsonb;
  end;
end
$function$;

create or replace function public.server_update_public_tournament_registration_edit(
  p_tournament_id text, p_registration_id text, p_expected_updated_at timestamptz,
  p_expected_selection_versions jsonb, p_registration_patch jsonb, p_selections jsonb
)
returns jsonb language plpgsql security invoker set search_path = ''
as $function$
declare
  v_original public.tournament_registration_selections%rowtype;
  v_scope text[];
  v_desired jsonb;
  v_baseline_selections jsonb := '[]'::jsonb;
  v_transfers jsonb := '[]'::jsonb;
  v_completed jsonb := '[]'::jsonb;
  v_changed_partner_registrations text[] := array[]::text[];
  v_result jsonb;
  v_transfer_result jsonb;
  v_version timestamptz;
  v_key text;
begin
  if jsonb_typeof(p_selections) is distinct from 'array' then
    return private.public_edit_before_division_transfer(p_tournament_id,p_registration_id,
      p_expected_updated_at,p_expected_selection_versions,p_registration_patch,p_selections);
  end if;
  -- Discover every parent and selection that this save can move before taking
  -- locks, so two partners editing the same pair cannot invert lock order.
  select array_agg(distinct s.id order by s.id) into v_scope
    from public.tournament_registration_selections s
    where s.registration_id = p_registration_id or s.registration_id in (
      select case when l.registration1_id = p_registration_id then l.registration2_id else l.registration1_id end
      from public.tournament_registration_team_links l
      where l.tournament_id::text = p_tournament_id and l.status in ('CONFIRMED','ADMIN_CONFIRMED')
        and p_registration_id in (l.registration1_id,l.registration2_id));
  if coalesce(cardinality(v_scope),0) > 0 then
    perform private.lock_tournament_registration_selection_scope(v_scope);
  end if;
  for v_desired in select value from jsonb_array_elements(p_selections) loop
    select * into v_original from public.tournament_registration_selections
      where id = v_desired->>'id' and registration_id = p_registration_id and tournament_id = p_tournament_id;
    if v_original.id is not null and v_original.event_option_id is distinct from v_desired->>'event_option_id'
      and exists (select 1 from public.tournament_registration_team_links l
        where l.status in ('CONFIRMED','ADMIN_CONFIRMED') and v_original.id in (l.selection1_id,l.selection2_id)) then
      foreach v_key in array array['partner_mode','partner_name','partner_email','partner_phone','partner_dupr_id',
        'partner_skill','partner_age','partner_gender'] loop
        if v_desired->v_key is distinct from to_jsonb(v_original)->v_key then
          return jsonb_build_object('ok',false,'code','REGISTRATION_RELATIONSHIP_LOCKED');
        end if;
      end loop;
      if not exists (select 1 from public.tournament_event_options
        where id = v_desired->>'event_option_id' and registration_day_id = v_desired->>'registration_day_id') then
        return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','INVALID_DIVISION_TARGET');
      end if;
      if not exists (select 1 from public.tournament_registrations r
        where r.id = v_desired->'division_transfer_context'->>'partner_registration_id'
          and r.updated_at = (v_desired->'division_transfer_context'->>'partner_registration_updated_at')::timestamptz) then
        return jsonb_build_object('ok',false,'code','SELECTION_WRITE_CONFLICT','reason','STALE_TEAM_VERSION');
      end if;
      v_transfers := v_transfers || jsonb_build_array(v_desired);
      v_desired := v_desired || jsonb_build_object('event_option_id',v_original.event_option_id,
        'registration_day_id',v_original.registration_day_id);
    end if;
    v_baseline_selections := v_baseline_selections || jsonb_build_array(v_desired - 'division_transfer_context');
  end loop;
  if jsonb_array_length(v_transfers) > 0 then
    -- Lock the union before baseline triggers acquire any SHARE event locks.
    -- This avoids a SHARE-to-UPDATE upgrade deadlock between distinct teams
    -- leaving the same source, including saves that touch multiple families.
    perform 1 from public.tournament_event_options event
      where event.id in (
        select value->>'event_option_id' from jsonb_array_elements(p_selections)
        union
        select value->>'event_option_id' from jsonb_array_elements(v_baseline_selections)
      ) order by event.id for update;
  end if;
  begin
    v_result := private.public_edit_before_division_transfer(p_tournament_id,p_registration_id,
      p_expected_updated_at,p_expected_selection_versions,p_registration_patch,v_baseline_selections);
    if (v_result->>'ok') is distinct from 'true' then
      return v_result;
    end if;
    for v_desired in select value from jsonb_array_elements(v_transfers) loop
      -- Every original partner version was checked before any writes. A
      -- previous transfer in this same save may already have advanced it.
      if (v_desired->'division_transfer_context'->>'partner_registration_id') = any(v_changed_partner_registrations) then
        select updated_at into v_version from public.tournament_registrations
          where id = v_desired->'division_transfer_context'->>'partner_registration_id';
        v_desired := jsonb_set(v_desired,'{division_transfer_context,partner_registration_updated_at}',to_jsonb(v_version));
      end if;
      select updated_at into v_version from public.tournament_registration_selections where id = v_desired->>'id';
      v_transfer_result := private.transfer_tournament_registration_team_division(p_tournament_id,
        v_desired->>'id',v_desired->>'event_option_id',v_version,v_desired->'division_transfer_context');
      if (v_transfer_result->>'ok') is distinct from 'true' then
        raise exception using errcode='PT409',message=v_transfer_result::text;
      end if;
      v_completed := v_completed || jsonb_build_array(v_transfer_result->'team_transfer');
      v_changed_partner_registrations := array_append(v_changed_partner_registrations,
        v_desired->'division_transfer_context'->>'partner_registration_id');
    end loop;
    select updated_at into v_version from public.tournament_registrations where id = p_registration_id;
    return v_result || jsonb_build_object('updated_at',v_version,'team_transfers',v_completed);
  exception when sqlstate 'PT409' then
    -- The subtransaction rolls back the full save and all earlier team moves.
    return sqlerrm::jsonb;
  end;
end
$function$;

revoke all on function public.admin_update_tournament_registration_selection(text,text,timestamptz,jsonb) from public, anon, authenticated;
grant execute on function public.admin_update_tournament_registration_selection(text,text,timestamptz,jsonb) to service_role;
revoke all on function public.server_update_public_tournament_registration_edit(text,text,timestamptz,jsonb,jsonb,jsonb) from public, anon, authenticated;
grant execute on function public.server_update_public_tournament_registration_edit(text,text,timestamptz,jsonb,jsonb,jsonb) to service_role;

-- Admission writes participate in the same target event lock as transfers.
create or replace function private.guard_tournament_registration_selection_identity()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration_tournament_id text;
  v_target_event public.tournament_event_options%rowtype;
  v_target_family text;
begin
  if tg_op = 'INSERT' then
    if nullif(btrim(new.id), '') is null
       or nullif(btrim(new.tournament_id), '') is null
       or nullif(btrim(new.registration_id), '') is null
       or nullif(btrim(new.event_option_id), '') is null
       or nullif(btrim(new.registration_day_id), '') is null then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_INVALID_TARGET: selection identity, tournament, registration, day, and event are required.';
    end if;

    -- New rows do not exist yet, so acquire the same parent-registration lock
    -- directly before the per-selection lock used by relationship mutations.
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-registration:' || btrim(new.registration_id),
        0
      )
    );
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:tournament-selection:' || btrim(new.id),
        0
      )
    );

    select registration.tournament_id
    into v_registration_tournament_id
    from public.tournament_registrations as registration
    where registration.id = btrim(new.registration_id)
    for update;

    if not found
       or v_registration_tournament_id <> btrim(new.tournament_id)
       or not exists (
         select 1
         from public.tournaments as tournament
         where tournament.id::text = btrim(new.tournament_id)
       ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_INVALID_TARGET: selection does not match an existing tournament registration.';
    end if;
  else
    if new.id is distinct from old.id
       or new.tournament_id is distinct from old.tournament_id
       or new.registration_id is distinct from old.registration_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_SELECTION_IDENTITY_IMMUTABLE: selection identity columns cannot be changed.';
    end if;

    if (
      new.event_option_id is distinct from old.event_option_id
      or new.registration_day_id is distinct from old.registration_day_id
    ) and coalesce(
      pg_catalog.current_setting('jupr.selection_edit_rpc', true),
      ''
    ) <> 'on' then
      raise exception using
        errcode = '42501',
        message = 'JUPR_SELECTION_EVENT_UPDATE_REQUIRES_RPC: use admin_update_tournament_registration_selection.';
    end if;
  end if;

  select event_option.*
  into v_target_event
  from public.tournament_event_options as event_option
  where event_option.id = new.event_option_id
    and event_option.tournament_id = new.tournament_id
  for share;

  if not found
     or new.registration_day_id is distinct from v_target_event.registration_day_id then
    raise exception using
      errcode = '23514',
      message = 'JUPR_SELECTION_INVALID_TARGET: selection event, day, and tournament are inconsistent.';
  end if;

  v_target_family := lower(
    regexp_replace(
      btrim(
        coalesce(
          nullif(v_target_event.event_family_label, ''),
          nullif(v_target_event.label, ''),
          'Event'
        )
      ),
      '\s+',
      ' ',
      'g'
    )
  );

  if exists (
    select 1
    from public.tournament_registration_selections as sibling
    join public.tournament_event_options as sibling_event
      on sibling_event.id = sibling.event_option_id
    where sibling.registration_id = new.registration_id
      and sibling.id <> new.id
      and sibling.registration_day_id = new.registration_day_id
      and lower(
        regexp_replace(
          btrim(
            coalesce(
              nullif(sibling_event.event_family_label, ''),
              nullif(sibling_event.label, ''),
              'Event'
            )
          ),
          '\s+',
          ' ',
          'g'
        )
      ) = v_target_family
  ) then
    raise exception using
      errcode = '23505',
      message = 'JUPR_SELECTION_WRITE_CONFLICT: duplicate event family for this registration day.';
  end if;

  return new;
end
$function$;

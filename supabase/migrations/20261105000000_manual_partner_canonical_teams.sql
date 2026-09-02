-- Canonicalize a submitted manual tournament partner into a player,
-- registration, selection, and confirmed two-person team in one transaction.
--
-- The trigger covers public registration (including commerce), secure public
-- edits, and the admin registration editor. Free-text partner columns remain
-- as the submitted eligibility snapshot, but they are no longer the team
-- authority once the write succeeds.

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

revoke all on function private.canonicalize_tournament_manual_partner_team_v1() from public;

drop trigger if exists trg_canonicalize_tournament_manual_partner_team
  on public.tournament_registration_selections;
create trigger trg_canonicalize_tournament_manual_partner_team
after insert or update of
  partner_mode,
  partner_name,
  partner_email,
  partner_phone,
  partner_dupr_id,
  partner_skill,
  partner_age,
  partner_gender,
  event_option_id,
  registration_day_id
on public.tournament_registration_selections
for each row
execute function private.canonicalize_tournament_manual_partner_team_v1();

-- Non-commerce public intake previously issued three separate Data API writes.
-- Use one RPC so registration + selections + trigger-created partner teams
-- commit or roll back together.
create or replace function public.create_tournament_registration_canonical_v1(
  p_registration jsonb,
  p_selections jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration public.tournament_registrations%rowtype;
  v_selection_count integer := 0;
begin
  if pg_catalog.jsonb_typeof(p_registration) <> 'object' then
    raise exception 'JUPR_TOURNAMENT_REGISTRATION_INVALID: registration must be an object.';
  end if;
  if p_selections is null or pg_catalog.jsonb_typeof(p_selections) <> 'array' then
    raise exception 'JUPR_TOURNAMENT_REGISTRATION_INVALID: selections must be an array.';
  end if;
  if pg_catalog.jsonb_array_length(p_selections) < 1
     or pg_catalog.jsonb_array_length(p_selections) > 64 then
    raise exception 'JUPR_TOURNAMENT_REGISTRATION_INVALID: select between 1 and 64 events.';
  end if;

  begin
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
    age_bracket,
    gender,
    notes,
    wants_partner_board_contact
  ) values (
    p_registration->>'id',
    p_registration->>'tournament_id',
    (p_registration->>'submitted_at')::timestamptz,
    (p_registration->>'updated_at')::timestamptz,
    p_registration->>'status',
    p_registration->>'payment_status',
    nullif(p_registration->>'first_name', ''),
    nullif(p_registration->>'last_name', ''),
    p_registration->>'display_name',
    pg_catalog.lower(p_registration->>'email'),
    nullif(p_registration->>'phone', ''),
    nullif(p_registration->>'player_id', '')::integer,
    nullif(p_registration->>'dupr_id', ''),
    nullif(p_registration->>'doubles_skill', '')::numeric,
    nullif(p_registration->>'singles_skill', '')::numeric,
    nullif(p_registration->>'age', '')::integer,
    nullif(p_registration->>'age_bracket', ''),
    nullif(p_registration->>'gender', ''),
    nullif(p_registration->>'notes', ''),
    coalesce((p_registration->>'wants_partner_board_contact')::boolean, false)
  )
  returning * into v_registration;
  exception
    when unique_violation then
      raise exception 'JUPR_TOURNAMENT_REGISTRATION_DUPLICATE: a registration already exists for this email or player.';
  end;

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
    row.id,
    v_registration.tournament_id::text,
    v_registration.id,
    row.registration_day_id,
    row.event_option_id,
    row.partner_mode,
    row.partner_name,
    row.partner_email,
    row.partner_phone,
    row.partner_dupr_id,
    row.partner_skill,
    row.partner_age,
    row.partner_gender,
    row.partner_note,
    row.show_on_partner_board,
    row.sort_order,
    row.created_at,
    row.updated_at
  from pg_catalog.jsonb_populate_recordset(
    null::public.tournament_registration_selections,
    p_selections
  ) as row;
  get diagnostics v_selection_count = row_count;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'registration_id', v_registration.id,
    'submitted_at', v_registration.submitted_at,
    'updated_at', v_registration.updated_at,
    'selection_count', v_selection_count
  );
end
$function$;

revoke all on function public.create_tournament_registration_canonical_v1(jsonb, jsonb) from public;
revoke all on function public.create_tournament_registration_canonical_v1(jsonb, jsonb) from anon, authenticated;
grant execute on function public.create_tournament_registration_canonical_v1(jsonb, jsonb) to service_role;

notify pgrst, 'reload schema';

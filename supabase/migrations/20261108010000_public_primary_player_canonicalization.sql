-- Canonicalize the primary registrant inside the same server-only transaction
-- that creates the registration. A public player_id is never trusted. An
-- unverified email may create a new club player only when it has no prior club
-- identity collision; collisions remain unlinked and are queued for staff
-- reconciliation without exposing which identity matched.

create extension if not exists pgcrypto with schema extensions;

create table if not exists public.tournament_primary_player_reconciliation (
  registration_id text primary key
    references public.tournament_registrations(id) on delete cascade,
  tournament_id uuid not null
    references public.tournaments(id) on delete cascade,
  club_id text not null,
  identity_hash text not null,
  state text not null check (
    state in (
      'CREATED',
      'CREATED_UNRATED',
      'EXISTING_REGISTRATION',
      'STAFF_RECONCILIATION_REQUIRED'
    )
  ),
  reason_code text,
  created_player_id integer references public.players(id) on delete set null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp()
);

create index if not exists tournament_primary_player_reconciliation_queue_idx
  on public.tournament_primary_player_reconciliation (club_id, state, created_at)
  where state = 'STAFF_RECONCILIATION_REQUIRED';
create index if not exists tournament_primary_player_reconciliation_tournament_idx
  on public.tournament_primary_player_reconciliation (tournament_id);
create index if not exists tournament_primary_player_reconciliation_created_player_idx
  on public.tournament_primary_player_reconciliation (created_player_id);

alter table public.tournament_primary_player_reconciliation enable row level security;
alter table public.tournament_primary_player_reconciliation force row level security;
revoke all on table public.tournament_primary_player_reconciliation
  from public, anon, authenticated, service_role;
grant select, insert, update on table public.tournament_primary_player_reconciliation
  to service_role;

comment on table public.tournament_primary_player_reconciliation is
  'Server-only primary-player linkage evidence. Stores a one-way identity hash, never a submitted email or public candidate id.';

create table if not exists public.tournament_registration_player_projection_reconciliation (
  projection_kind text not null check (
    projection_kind in (
      'TEAM_MEMBER',
      'TEAM_LINK_PLAYER1',
      'TEAM_LINK_PLAYER2'
    )
  ),
  relationship_id text not null,
  tournament_id uuid not null
    references public.tournaments(id) on delete cascade,
  registration_id text not null
    references public.tournament_registrations(id) on delete cascade,
  selection_id text not null
    references public.tournament_registration_selections(id) on delete cascade,
  registration_player_id integer null
    references public.players(id) on delete set null,
  reason_code text not null check (
    reason_code in (
      'FINALIZED_REGISTRATION_CLOSE_REVIEW',
      'REGISTRATION_PLAYER_CLUB_MISMATCH'
    )
  ),
  state text not null default 'PENDING' check (
    state in ('PENDING', 'RESOLVED', 'DISMISSED')
  ),
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  updated_at timestamptz not null default pg_catalog.clock_timestamp(),
  primary key (projection_kind, relationship_id)
);

create index if not exists tournament_player_projection_reconciliation_tournament_idx
  on public.tournament_registration_player_projection_reconciliation (
    tournament_id, state
  );
create index if not exists tournament_player_projection_reconciliation_registration_idx
  on public.tournament_registration_player_projection_reconciliation (
    registration_id
  );
create index if not exists tournament_player_projection_reconciliation_selection_idx
  on public.tournament_registration_player_projection_reconciliation (
    selection_id
  );
create index if not exists tournament_player_projection_reconciliation_player_idx
  on public.tournament_registration_player_projection_reconciliation (
    registration_player_id
  );

alter table public.tournament_registration_player_projection_reconciliation
  enable row level security;
alter table public.tournament_registration_player_projection_reconciliation
  force row level security;
revoke all on table public.tournament_registration_player_projection_reconciliation
  from public, anon, authenticated, service_role;
grant select, insert, update
  on table public.tournament_registration_player_projection_reconciliation
  to service_role;

comment on table public.tournament_registration_player_projection_reconciliation is
  'Server-only queue for null tournament team player projections that cannot be safely filled because finalized rating evidence is immutable or the registration player belongs to another club.';

create or replace function private.resolve_public_tournament_primary_player_v1(
  p_club_id text,
  p_tournament_id text,
  p_registration jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_registration_id text := nullif(pg_catalog.btrim(p_registration ->> 'id'), '');
  v_email text := pg_catalog.lower(nullif(pg_catalog.btrim(p_registration ->> 'email'), ''));
  v_display_name text := nullif(pg_catalog.btrim(coalesce(
    p_registration ->> 'display_name',
    pg_catalog.concat_ws(
      ' ',
      nullif(pg_catalog.btrim(p_registration ->> 'first_name'), ''),
      nullif(pg_catalog.btrim(p_registration ->> 'last_name'), '')
    )
  )), '');
  v_normalized_name text;
  v_email_lock bigint;
  v_name_lock bigint;
  v_existing_registration public.tournament_registrations%rowtype;
  v_existing_reconciliation public.tournament_primary_player_reconciliation%rowtype;
  v_prior_identity_count integer := 0;
  v_skill numeric;
  v_player public.players%rowtype;
  v_state text;
begin
  if v_registration_id is null or v_email is null or v_display_name is null then
    raise exception 'JUPR_PRIMARY_PLAYER_IDENTITY_INVALID: name and email are required.';
  end if;
  if not exists (
    select 1
      from public.tournaments as tournament
     where tournament.id::text = p_tournament_id
       and tournament.club_id::text = p_club_id
  ) then
    raise exception 'JUPR_PRIMARY_PLAYER_TOURNAMENT_MISMATCH: tournament does not belong to this club.';
  end if;

  v_normalized_name := pg_catalog.lower(pg_catalog.btrim(v_display_name));
  v_email_lock := pg_catalog.hashtextextended(
    'public-primary-player-email:' || p_club_id || ':' || v_email,
    0
  );
  v_name_lock := pg_catalog.hashtextextended(
    'public-primary-player-name:' || p_club_id || ':' || v_normalized_name,
    0
  );
  -- Every resolver takes both identity locks in numeric order. This serializes
  -- same-email and same-normalized-name intake without introducing a lock-order
  -- deadlock when two requests collide on different identity dimensions.
  perform pg_catalog.pg_advisory_xact_lock(case
    when v_email_lock <= v_name_lock then v_email_lock else v_name_lock
  end);
  if v_email_lock <> v_name_lock then
    perform pg_catalog.pg_advisory_xact_lock(case
      when v_email_lock > v_name_lock then v_email_lock else v_name_lock
    end);
  end if;

  -- A durable same-id row is trusted idempotency evidence. This is the only
  -- existing player link accepted without a signed registration token.
  select registration.*
    into v_existing_registration
    from public.tournament_registrations as registration
   where registration.id = v_registration_id
     and registration.tournament_id::text = p_tournament_id
   limit 1;
  if v_existing_registration.id is not null
     and v_existing_registration.player_id is not null then
    select reconciliation.*
      into v_existing_reconciliation
      from public.tournament_primary_player_reconciliation as reconciliation
     where reconciliation.registration_id = v_registration_id
       and reconciliation.tournament_id::text = p_tournament_id
       and reconciliation.club_id = p_club_id
     limit 1;
    if v_existing_reconciliation.registration_id is not null
       and v_existing_reconciliation.created_player_id = v_existing_registration.player_id
       and v_existing_reconciliation.state in ('CREATED', 'CREATED_UNRATED') then
      return pg_catalog.jsonb_build_object(
        'player_id', v_existing_registration.player_id,
        'state', v_existing_reconciliation.state,
        'reason_code', v_existing_reconciliation.reason_code
      );
    end if;
    return pg_catalog.jsonb_build_object(
      'player_id', v_existing_registration.player_id,
      'state', 'EXISTING_REGISTRATION',
      'reason_code', null
    );
  end if;

  -- Email supplied during initial intake is not verified. Any prior same-email
  -- identity in this club is therefore a collision, not authority to link.
  select pg_catalog.count(*)::integer
    into v_prior_identity_count
    from public.tournament_registrations as registration
    join public.tournaments as tournament
      on tournament.id::text = registration.tournament_id::text
   where tournament.club_id::text = p_club_id
     and pg_catalog.lower(pg_catalog.btrim(registration.email)) = v_email
     and registration.id <> v_registration_id;

  if v_prior_identity_count > 0 then
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EMAIL_COLLISION'
    );
  end if;

  -- A public display name is also unverified. Match the historical backfill's
  -- normalized-name rule and leave any collision for staff; never let the
  -- players uniqueness constraint turn a valid registration into a 500.
  if exists (
    select 1
      from public.players as player
     where player.club_id = p_club_id
       and pg_catalog.lower(pg_catalog.btrim(player.name)) = v_normalized_name
  ) then
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EXISTING_PLAYER_NAME_COLLISION'
    );
  end if;

  begin
    v_skill := nullif(p_registration ->> 'doubles_skill', '')::numeric;
  exception when others then
    v_skill := null;
  end;
  if v_skill is null or v_skill < 1 or v_skill > 7 then
    begin
      v_skill := nullif(p_registration ->> 'singles_skill', '')::numeric;
    exception when others then
      v_skill := null;
    end;
  end if;
  if v_skill is null or v_skill < 1 or v_skill > 7 then
    -- 3.0 is an explicit temporary baseline, recorded for staff reconciliation;
    -- it is not inferred from an unverified public player candidate.
    v_skill := 3.0;
    v_state := 'CREATED_UNRATED';
  else
    v_state := 'CREATED';
  end if;

  begin
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
      p_club_id,
      v_display_name,
      v_skill * 400.0,
      v_skill * 400.0,
      0,
      0,
      0,
      true,
      null,
      null
    )
    returning * into v_player;
  exception when unique_violation then
    -- Defensive fallback for a concurrent/manual player writer that does not
    -- participate in the advisory-lock protocol.
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EXISTING_PLAYER_NAME_COLLISION'
    );
  end;

  return pg_catalog.jsonb_build_object(
    'player_id', v_player.id,
    'state', v_state,
    'reason_code', case
      when v_state = 'CREATED_UNRATED' then 'TEMPORARY_3_0_BASELINE'
      else null
    end
  );
end
$function$;

revoke all on function private.resolve_public_tournament_primary_player_v1(
  text,
  text,
  jsonb
) from public, anon, authenticated;
grant execute on function private.resolve_public_tournament_primary_player_v1(
  text,
  text,
  jsonb
) to service_role;

-- Bring historical unlinked primary registrants under the same canonical
-- policy. Unique, non-colliding identities receive a new independent player;
-- ambiguous names/emails remain unlinked in the private staff queue. No
-- existing player is ever selected from an unverified public email.
do $historical_primary_player_backfill$
declare
  v_registration record;
  v_resolution jsonb;
  v_display_name text;
  v_state text;
  v_reason_code text;
begin
  lock table public.tournament_registrations in share row exclusive mode;
  lock table public.players in share row exclusive mode;

  for v_registration in
    select registration.*, tournament.club_id::text as resolved_club_id
      from public.tournament_registrations as registration
      join public.tournaments as tournament
        on tournament.id::text = registration.tournament_id::text
     where registration.player_id is null
       and nullif(pg_catalog.btrim(registration.email), '') is not null
     order by registration.id
     for update of registration
  loop
    v_display_name := nullif(pg_catalog.btrim(coalesce(
      nullif(v_registration.display_name, ''),
      pg_catalog.concat_ws(
        ' ',
        nullif(pg_catalog.btrim(v_registration.first_name), ''),
        nullif(pg_catalog.btrim(v_registration.last_name), '')
      )
    )), '');

    if exists (
      select 1
        from public.tournament_rating_eligibility_reviews as review
       where (
         review.registration_id::text = v_registration.id
         or review.partner_registration_id::text = v_registration.id
       )
         and review.review_phase = 'REGISTRATION_CLOSE'
         and review.finalized_at is not null
    ) then
      -- The registration identity guard would reject this update. Queue it
      -- before invoking the resolver so no unused player is created.
      v_resolution := pg_catalog.jsonb_build_object(
        'player_id', null,
        'state', 'STAFF_RECONCILIATION_REQUIRED',
        'reason_code', 'FINALIZED_REGISTRATION_CLOSE_REVIEW'
      );
    elsif v_display_name is null then
      v_resolution := pg_catalog.jsonb_build_object(
        'player_id', null,
        'state', 'STAFF_RECONCILIATION_REQUIRED',
        'reason_code', 'INVALID_DISPLAY_NAME'
      );
    elsif exists (
      select 1
        from public.players as player
       where player.club_id = v_registration.resolved_club_id
         and pg_catalog.lower(pg_catalog.btrim(player.name)) =
             pg_catalog.lower(v_display_name)
    ) then
      v_resolution := pg_catalog.jsonb_build_object(
        'player_id', null,
        'state', 'STAFF_RECONCILIATION_REQUIRED',
        'reason_code', 'EXISTING_PLAYER_NAME_COLLISION'
      );
    else
      v_resolution := private.resolve_public_tournament_primary_player_v1(
        v_registration.resolved_club_id,
        v_registration.tournament_id::text,
        pg_catalog.to_jsonb(v_registration) - 'resolved_club_id' - 'player_id'
      );
    end if;

    if nullif(v_resolution ->> 'player_id', '') is not null then
      update public.tournament_registrations as registration
         set player_id = (v_resolution ->> 'player_id')::integer,
             updated_at = pg_catalog.clock_timestamp()
       where registration.id = v_registration.id
         and registration.tournament_id = v_registration.tournament_id
         and registration.player_id is null;
    end if;

    v_state := coalesce(
      nullif(v_resolution ->> 'state', ''),
      'STAFF_RECONCILIATION_REQUIRED'
    );
    v_reason_code := nullif(v_resolution ->> 'reason_code', '');
    insert into public.tournament_primary_player_reconciliation (
      registration_id,
      tournament_id,
      club_id,
      identity_hash,
      state,
      reason_code,
      created_player_id,
      updated_at
    ) values (
      v_registration.id,
      v_registration.tournament_id::uuid,
      v_registration.resolved_club_id,
      pg_catalog.encode(
        extensions.digest(
          v_registration.resolved_club_id || ':' ||
          pg_catalog.lower(pg_catalog.btrim(v_registration.email)),
          'sha256'
        ),
        'hex'
      ),
      v_state,
      v_reason_code,
      nullif(v_resolution ->> 'player_id', '')::integer,
      pg_catalog.clock_timestamp()
    )
    on conflict (registration_id) do update set
      state = excluded.state,
      reason_code = excluded.reason_code,
      created_player_id = excluded.created_player_id,
      updated_at = excluded.updated_at;
  end loop;
end
$historical_primary_player_backfill$;

-- Registrations linked above are the canonical player identity for their
-- same-tournament partner-team projections. Repair only live null projections;
-- never rewrite an existing member/link player, even when it disagrees.
-- Finalized relationship reviews are immutable, and cross-club registration
-- player links are not trusted as projection sources; queue both for staff.
with projection_candidates (
  projection_kind,
  relationship_id,
  tournament_id,
  registration_id,
  selection_id,
  trigger_selection_ids,
  registration_player_id
) as (
  select
    'TEAM_MEMBER',
    member.id,
    member.tournament_id,
    member.registration_id,
    member.selection_id,
    array[member.selection_id::text],
    registration.player_id
  from public.tournament_registration_team_members as member
  join public.tournament_registrations as registration
    on registration.id = member.registration_id
   and registration.tournament_id = member.tournament_id::text
  where member.player_id is null
    and member.status = 'ACTIVE'
    and registration.player_id is not null

  union all

  select
    'TEAM_LINK_PLAYER1',
    link.id,
    link.tournament_id,
    link.registration1_id,
    link.selection1_id,
    array[link.selection1_id::text, link.selection2_id::text],
    registration.player_id
  from public.tournament_registration_team_links as link
  join public.tournament_registrations as registration
    on registration.id = link.registration1_id
   and registration.tournament_id = link.tournament_id::text
  where link.player1_id is null
    and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
    and registration.player_id is not null

  union all

  select
    'TEAM_LINK_PLAYER2',
    link.id,
    link.tournament_id,
    link.registration2_id,
    link.selection2_id,
    array[link.selection1_id::text, link.selection2_id::text],
    registration.player_id
  from public.tournament_registration_team_links as link
  join public.tournament_registrations as registration
    on registration.id = link.registration2_id
   and registration.tournament_id = link.tournament_id::text
  where link.player2_id is null
    and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
    and registration.player_id is not null
)
insert into public.tournament_registration_player_projection_reconciliation (
  projection_kind,
  relationship_id,
  tournament_id,
  registration_id,
  selection_id,
  registration_player_id,
  reason_code,
  state,
  updated_at
)
select
  candidate.projection_kind,
  candidate.relationship_id,
  candidate.tournament_id,
  candidate.registration_id,
  candidate.selection_id,
  candidate.registration_player_id,
  case
    when player.id is null
      or player.club_id is distinct from tournament.club_id
      then 'REGISTRATION_PLAYER_CLUB_MISMATCH'
    else 'FINALIZED_REGISTRATION_CLOSE_REVIEW'
  end,
  'PENDING',
  pg_catalog.clock_timestamp()
from projection_candidates as candidate
join public.tournaments as tournament
  on tournament.id = candidate.tournament_id
left join public.players as player
  on player.id = candidate.registration_player_id
where player.id is null
   or player.club_id is distinct from tournament.club_id
   or exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text = any(candidate.trigger_selection_ids)
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   )
on conflict (projection_kind, relationship_id) do update set
  tournament_id = excluded.tournament_id,
  registration_id = excluded.registration_id,
  selection_id = excluded.selection_id,
  registration_player_id = excluded.registration_player_id,
  reason_code = excluded.reason_code,
  state = 'PENDING',
  updated_at = excluded.updated_at;

update public.tournament_registration_team_members as member
   set player_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where member.player_id is null
   and member.status = 'ACTIVE'
   and registration.id = member.registration_id
   and registration.tournament_id = member.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text = member.selection_id::text
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

update public.tournament_registration_team_links as link
   set player1_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where link.player1_id is null
   and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
   and registration.id = link.registration1_id
   and registration.tournament_id = link.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text in (
        link.selection1_id::text,
        link.selection2_id::text
      )
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

update public.tournament_registration_team_links as link
   set player2_id = registration.player_id
  from public.tournament_registrations as registration
  join public.tournaments as tournament
    on tournament.id::text = registration.tournament_id
  join public.players as player
    on player.id = registration.player_id
   and player.club_id = tournament.club_id
 where link.player2_id is null
   and link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
   and registration.id = link.registration2_id
   and registration.tournament_id = link.tournament_id::text
   and registration.player_id is not null
   and not exists (
     select 1
       from public.tournament_rating_eligibility_reviews as review
      where review.selection_id::text in (
        link.selection1_id::text,
        link.selection2_id::text
      )
        and review.review_phase = 'REGISTRATION_CLOSE'
        and review.finalized_at is not null
   );

-- Preserve the already-hardened implementations as private-by-convention raw
-- operations. The wrappers below resolve primary identity, then delegate in the
-- same database transaction.
alter function public.create_tournament_registration_canonical_v1(jsonb, jsonb)
  rename to create_tournament_registration_canonical_unlinked_v1;

revoke all on function public.create_tournament_registration_canonical_unlinked_v1(
  jsonb,
  jsonb
) from public, anon, authenticated;
grant execute on function public.create_tournament_registration_canonical_unlinked_v1(
  jsonb,
  jsonb
) to service_role;

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
  v_tournament_id text := nullif(pg_catalog.btrim(p_registration ->> 'tournament_id'), '');
  v_club_id text;
  v_resolution jsonb;
  v_registration jsonb;
  v_result jsonb;
begin
  select tournament.club_id::text
    into v_club_id
    from public.tournaments as tournament
   where tournament.id::text = v_tournament_id;
  if v_club_id is null then
    raise exception 'JUPR_PRIMARY_PLAYER_TOURNAMENT_MISMATCH: tournament was not found.';
  end if;

  v_resolution := private.resolve_public_tournament_primary_player_v1(
    v_club_id,
    v_tournament_id,
    p_registration - 'player_id'
  );
  v_registration := pg_catalog.jsonb_set(
    p_registration - 'player_id',
    '{player_id}',
    coalesce(v_resolution -> 'player_id', 'null'::jsonb),
    true
  );
  v_result := public.create_tournament_registration_canonical_unlinked_v1(
    v_registration,
    p_selections
  );

  insert into public.tournament_primary_player_reconciliation (
    registration_id,
    tournament_id,
    club_id,
    identity_hash,
    state,
    reason_code,
    created_player_id,
    updated_at
  ) values (
    v_result ->> 'registration_id',
    v_tournament_id::uuid,
    v_club_id,
    pg_catalog.encode(
      extensions.digest(
        v_club_id || ':' || pg_catalog.lower(pg_catalog.btrim(p_registration ->> 'email')),
        'sha256'
      ),
      'hex'
    ),
    v_resolution ->> 'state',
    nullif(v_resolution ->> 'reason_code', ''),
    nullif(v_resolution ->> 'player_id', '')::integer,
    pg_catalog.clock_timestamp()
  )
  on conflict (registration_id) do update set
    state = excluded.state,
    reason_code = excluded.reason_code,
    created_player_id = excluded.created_player_id,
    updated_at = excluded.updated_at;

  return v_result;
end
$function$;

revoke all on function public.create_tournament_registration_canonical_v1(
  jsonb,
  jsonb
) from public, anon, authenticated;
grant execute on function public.create_tournament_registration_canonical_v1(
  jsonb,
  jsonb
) to service_role;

-- Admin creation is a separate trusted server path. It preserves an explicitly
-- selected player_id only after proving that player belongs to the tournament's
-- club. Public intake continues through the wrapper above, where player_id is
-- always stripped and independently resolved.
create or replace function public.create_admin_tournament_registration_canonical_v1(
  p_registration jsonb,
  p_selections jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_tournament_id text := nullif(
    pg_catalog.btrim(p_registration ->> 'tournament_id'), ''
  );
  v_club_id text;
  v_player_id integer;
begin
  select tournament.club_id::text
    into v_club_id
    from public.tournaments as tournament
   where tournament.id::text = v_tournament_id;
  if v_club_id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_ADMIN_REGISTRATION_TOURNAMENT_MISMATCH';
  end if;

  begin
    v_player_id := nullif(
      pg_catalog.btrim(p_registration ->> 'player_id'), ''
    )::integer;
  exception when others then
    raise exception using errcode = '22023',
      message = 'JUPR_ADMIN_REGISTRATION_PLAYER_MISMATCH';
  end;

  if v_player_id is not null and (
    v_player_id <= 0
    or not exists (
      select 1
        from public.players as player
       where player.id = v_player_id
         and player.club_id = v_club_id
    )
  ) then
    raise exception using errcode = '22023',
      message = 'JUPR_ADMIN_REGISTRATION_PLAYER_MISMATCH';
  end if;

  return public.create_tournament_registration_canonical_unlinked_v1(
    p_registration,
    p_selections
  );
end
$function$;

revoke all on function public.create_admin_tournament_registration_canonical_v1(
  jsonb,
  jsonb
) from public, anon, authenticated;
grant execute on function public.create_admin_tournament_registration_canonical_v1(
  jsonb,
  jsonb
) to service_role;

comment on function public.create_admin_tournament_registration_canonical_v1(
  jsonb,
  jsonb
) is
  'Server-only admin registration create. Preserves player_id only when it belongs to the tournament club; public intake must use the untrusted canonical wrapper.';

alter function public.server_create_public_tournament_registration_with_commerce(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) rename to server_create_public_tournament_registration_with_commerce_unlinked_v1;

revoke all on function public.server_create_public_tournament_registration_with_commerce_unlinked_v1(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.server_create_public_tournament_registration_with_commerce_unlinked_v1(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) to service_role;

create or replace function public.server_create_public_tournament_registration_with_commerce(
  p_club_id text,
  p_tournament_id uuid,
  p_registration jsonb,
  p_selections jsonb,
  p_quote_snapshot jsonb,
  p_operation_idempotency_key uuid,
  p_order_idempotency_key uuid,
  p_request_fingerprint text,
  p_actor_label text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_resolution jsonb;
  v_registration jsonb;
  v_result jsonb;
begin
  v_resolution := private.resolve_public_tournament_primary_player_v1(
    p_club_id,
    p_tournament_id::text,
    p_registration - 'player_id'
  );
  v_registration := pg_catalog.jsonb_set(
    p_registration - 'player_id',
    '{player_id}',
    coalesce(v_resolution -> 'player_id', 'null'::jsonb),
    true
  );
  v_result := public.server_create_public_tournament_registration_with_commerce_unlinked_v1(
    p_club_id,
    p_tournament_id,
    v_registration,
    p_selections,
    p_quote_snapshot,
    p_operation_idempotency_key,
    p_order_idempotency_key,
    p_request_fingerprint,
    p_actor_label,
    p_source
  );

  insert into public.tournament_primary_player_reconciliation (
    registration_id,
    tournament_id,
    club_id,
    identity_hash,
    state,
    reason_code,
    created_player_id,
    updated_at
  ) values (
    v_result ->> 'registration_id',
    p_tournament_id,
    p_club_id,
    pg_catalog.encode(
      extensions.digest(
        p_club_id || ':' || pg_catalog.lower(pg_catalog.btrim(p_registration ->> 'email')),
        'sha256'
      ),
      'hex'
    ),
    v_resolution ->> 'state',
    nullif(v_resolution ->> 'reason_code', ''),
    nullif(v_resolution ->> 'player_id', '')::integer,
    pg_catalog.clock_timestamp()
  )
  on conflict (registration_id) do update set
    state = excluded.state,
    reason_code = excluded.reason_code,
    created_player_id = excluded.created_player_id,
    updated_at = excluded.updated_at;

  return v_result;
end
$function$;

revoke all on function public.server_create_public_tournament_registration_with_commerce(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) from public, anon, authenticated;
grant execute on function public.server_create_public_tournament_registration_with_commerce(
  text,
  uuid,
  jsonb,
  jsonb,
  jsonb,
  uuid,
  uuid,
  text,
  text,
  text
) to service_role;

notify pgrst, 'reload schema';

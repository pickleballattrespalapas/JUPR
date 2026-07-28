-- Forward-only response-loss recovery for public fixed-partner registration.
--
-- A browser refresh creates a new transport idempotency key. This function
-- recovers an exact prior business request while rejecting changed requests
-- that involve a player who is already actively registered.

create index if not exists team_league_registration_recovery_idx
  on public.team_league_operations (
    club_id,
    league_name,
    operation_type,
    request_fingerprint,
    completed_at desc
  )
  where status = 'complete'
    and operation_type in ('public_team_signup', 'public_solo_signup');

create or replace function public.team_league_recover_public_registration_v1(
  p_club_id text,
  p_league_name text,
  p_request_fingerprint text,
  p_signup_type text,
  p_player_id bigint,
  p_partner_player_id bigint,
  p_invite_token_hash text,
  p_invite_expires_at timestamptz,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_league_name text := nullif(
    pg_catalog.left(pg_catalog.btrim(p_league_name), 120),
    ''
  );
  v_fingerprint text := pg_catalog.lower(
    pg_catalog.btrim(coalesce(p_request_fingerprint, ''))
  );
  v_signup_type text := pg_catalog.lower(
    pg_catalog.btrim(coalesce(p_signup_type, ''))
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'public_team_league_registration_recovery'
  );
  v_operation public.team_league_operations%rowtype;
  v_team public.team_league_teams%rowtype;
  v_result jsonb;
begin
  if v_club_id is null
     or v_league_name is null
     or v_fingerprint !~ '^[0-9a-f]{64}$'
     or v_signup_type not in ('team', 'solo')
     or p_player_id is null
     or (
       v_signup_type = 'team'
       and (
         p_partner_player_id is null
         or p_partner_player_id = p_player_id
         or p_invite_token_hash !~ '^[0-9a-f]{64}$'
         or p_invite_expires_at is null
       )
     ) then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SIGNUP_RECOVERY_INVALID';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(
      'pcs:team-league-registration:' || v_club_id || ':' ||
      v_league_name || ':' || v_signup_type || ':' ||
      least(
        p_player_id,
        coalesce(p_partner_player_id, p_player_id)
      )::text || ':' ||
      greatest(
        p_player_id,
        coalesce(p_partner_player_id, p_player_id)
      )::text,
      0
    )
  );

  if v_signup_type = 'team' then
    select operation.*
      into v_operation
      from public.team_league_operations as operation
      join public.team_league_teams as team
        on team.created_operation_id = operation.id
       and team.club_id = operation.club_id
       and team.league_name = operation.league_name
       and team.status in ('pending_partner', 'confirmed')
     where operation.club_id = v_club_id
       and operation.league_name = v_league_name
       and operation.operation_type = 'public_team_signup'
       and operation.request_fingerprint = v_fingerprint
       and operation.status = 'complete'
       and operation.result_json is not null
     order by operation.completed_at desc nulls last, operation.started_at desc
     limit 1
     for update of operation;
    if found then
      select team.*
        into strict v_team
        from public.team_league_teams as team
       where team.created_operation_id = v_operation.id
         and team.club_id = v_club_id
         and team.league_name = v_league_name
         and team.status in ('pending_partner', 'confirmed')
       for update;

      if v_team.status = 'confirmed' then
        return v_operation.result_json || pg_catalog.jsonb_build_object(
          'found', true,
          'idempotent', true,
          'recovered_by_business_identity', true,
          'current_status', 'confirmed',
          'invitation_send_required', false,
          'invitation_delivery_status', 'not_required'
        );
      end if;

      if v_team.invitation_delivery_status in (
        'dry_run',
        'staging_redirect',
        'sent'
      ) then
        return v_operation.result_json || pg_catalog.jsonb_build_object(
          'found', true,
          'idempotent', true,
          'recovered_by_business_identity', true,
          'invitation_send_required', false,
          'invitation_delivery_status', v_team.invitation_delivery_status
        );
      end if;

      if v_team.invitation_delivery_status = 'claimed'
         and v_team.invitation_claimed_at >
           pg_catalog.clock_timestamp() - interval '5 minutes' then
        return v_operation.result_json || pg_catalog.jsonb_build_object(
          'found', true,
          'idempotent', true,
          'recovered_by_business_identity', true,
          'invitation_send_required', false,
          'invitation_delivery_status', 'claimed',
          'invitation_delivery_in_progress', true
        );
      end if;

      update public.team_league_teams
         set partner_invite_token_hash = p_invite_token_hash,
             partner_invite_expires_at = p_invite_expires_at,
             invitation_delivery_status = 'pending',
             invitation_provider_message_id = null,
             invitation_delivered_at = null,
             invitation_claim_token = null,
             invitation_claimed_at = null,
             invitation_delivery_error = null,
             updated_at = pg_catalog.clock_timestamp()
       where id = v_team.id;

      insert into public.admin_activity_log (
        club_id,
        actor_email,
        actor_role,
        action_type,
        entity_type,
        entity_id,
        before_json,
        after_json,
        note,
        source_page,
        flagged_for_review
      ) values (
        v_club_id,
        v_team.captain_contact_email,
        'public',
        'team_league_registration_response_recovered',
        'team_league_team',
        v_team.id::text,
        pg_catalog.jsonb_build_object(
          'invitation_delivery_status', v_team.invitation_delivery_status,
          'invitation_delivery_attempts', v_team.invitation_delivery_attempts
        ),
        pg_catalog.jsonb_build_object(
          'invitation_delivery_status', 'pending',
          'invitation_delivery_attempts', v_team.invitation_delivery_attempts,
          'token_rotated', true
        ),
        'Recovered an exact registration after response loss.',
        v_source,
        false
      );

      return v_operation.result_json || pg_catalog.jsonb_build_object(
        'found', true,
        'idempotent', true,
        'recovered_by_business_identity', true,
        'invitation_send_required', true,
        'invitation_delivery_status', 'pending'
      );
    end if;
  else
    select operation.*
      into v_operation
      from public.team_league_operations as operation
      join public.team_league_solo_waitlist as waitlist
        on waitlist.created_operation_id = operation.id
       and waitlist.club_id = operation.club_id
       and waitlist.league_name = operation.league_name
       and waitlist.status = 'waiting'
     where operation.club_id = v_club_id
       and operation.league_name = v_league_name
       and operation.operation_type = 'public_solo_signup'
       and operation.request_fingerprint = v_fingerprint
       and operation.status = 'complete'
       and operation.result_json is not null
     order by operation.completed_at desc nulls last, operation.started_at desc
     limit 1
     for update of operation;
    if found then
      return v_operation.result_json || pg_catalog.jsonb_build_object(
        'found', true,
        'idempotent', true,
        'recovered_by_business_identity', true
      );
    end if;
  end if;

  if exists (
    select 1
      from public.team_league_teams as team
     where team.club_id = v_club_id
       and team.league_name = v_league_name
       and team.status in ('pending_partner', 'confirmed')
       and (
         team.captain_player_id = p_player_id
         or team.partner_player_id = p_player_id
         or (
           p_partner_player_id is not null
           and (
             team.captain_player_id = p_partner_player_id
             or team.partner_player_id = p_partner_player_id
           )
         )
       )
  ) or exists (
    select 1
      from public.team_league_solo_waitlist as waitlist
     where waitlist.club_id = v_club_id
       and waitlist.league_name = v_league_name
       and waitlist.status = 'waiting'
       and (
         waitlist.player_id = p_player_id
         or (
           p_partner_player_id is not null
           and waitlist.player_id = p_partner_player_id
         )
       )
  ) then
    raise exception using
      errcode = '23505',
      message = 'TEAM_LEAGUE_REGISTRATION_IDENTITY_CONFLICT';
  end if;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'found', false,
    'recovered_by_business_identity', false
  );
  return v_result;
end
$function$;

revoke all on function public.team_league_recover_public_registration_v1(
  text,
  text,
  text,
  text,
  bigint,
  bigint,
  text,
  timestamptz,
  text
) from public, anon, authenticated;

grant execute on function public.team_league_recover_public_registration_v1(
  text,
  text,
  text,
  text,
  bigint,
  bigint,
  text,
  timestamptz,
  text
) to service_role;

comment on function public.team_league_recover_public_registration_v1(
  text,
  text,
  text,
  text,
  bigint,
  bigint,
  text,
  timestamptz,
  text
) is
  'Recovers an exact active public registration after response loss and conflicts changed repeats without exposing invitation secrets.';

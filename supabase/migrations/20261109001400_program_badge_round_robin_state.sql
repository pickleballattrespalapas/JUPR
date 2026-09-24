-- Distinguish the locked evaluation record from live_sessions.state.
create or replace function public.admin_resolve_round_robin_v1(p_club_id text,p_actor_email text,p_actor_user_id uuid,
 p_actor_role text,p_operation_id uuid,p_payload jsonb) returns jsonb
language plpgsql security invoker set search_path='' as $$
declare v_state public.badge_program_state%rowtype; op public.badge_round_robin_operations%rowtype;
        tie jsonb; result jsonb; display_decision jsonb; winner bigint := (p_payload->>'winner_player_id')::bigint;
begin
  if p_actor_role not in ('administrator','super_admin','club_owner') or not exists (
    select 1 from public.admin_role_assignments a where a.club_id=p_club_id and a.email=lower(trim(p_actor_email))
      and a.role=p_actor_role and a.revoked_at is null and (a.expires_at is null or a.expires_at>now())
      and (a.user_id is null or a.user_id=p_actor_user_id)) then
    raise exception using errcode='42501',message='A current club administrator assignment is required.';
  end if;
  select * into v_state from public.badge_program_state where club_id=p_club_id for update;
  select * into op from public.badge_round_robin_operations where club_id=p_club_id and operation_id=p_operation_id;
  if found then
    if op.actor_user_id is distinct from p_actor_user_id or op.payload is distinct from p_payload then
      raise exception using errcode='40001',message='This save request was already used for another decision.';
    end if;
    return op.result;
  end if;
  if v_state.revision is distinct from v_state.applied_revision then
    raise exception using errcode='40001',message='Round-robin results changed. Refresh after the badge check finishes.';
  end if;
  select x into tie from jsonb_array_elements(v_state.pending_ties) x where x->>'source_key'=p_payload->>'source_key'
     and x->>'source_fingerprint'=p_payload->>'source_fingerprint';
  if tie is null or winner is null or not exists(select 1 from jsonb_array_elements(tie->'leaders') l where (l->>'player_id')::bigint=winner)
     or not exists(select 1 from public.players p where p.id=winner and p.club_id=p_club_id) then
    raise exception using errcode='40001',message='Choose a linked player among the current tied leaders. Refresh to load the current result.';
  end if;
  insert into public.badge_round_robin_decisions(club_id,source_key,source_fingerprint,winner_player_id,actor_user_id,actor_email,evidence)
  values(p_club_id,p_payload->>'source_key',p_payload->>'source_fingerprint',winner,p_actor_user_id,lower(trim(p_actor_email)),tie)
  on conflict(club_id,source_key) do update set source_fingerprint=excluded.source_fingerprint,winner_player_id=excluded.winner_player_id,
    actor_user_id=excluded.actor_user_id,actor_email=excluded.actor_email,decided_at=now(),evidence=excluded.evidence;
  -- Store a signed-by-results display decision with both durable forms of the
  -- session. The scorer ignores it automatically after any score/roster change.
  select jsonb_build_object('participant_id',l->>'participant_id','display_fingerprint',tie->>'display_fingerprint')
    into display_decision from jsonb_array_elements(tie->'leaders') l where (l->>'player_id')::bigint=winner;
  update public.live_events set raw_event_json=jsonb_set(raw_event_json,'{round_robin_winner}',display_decision),updated_at=now()
    where club_id=p_club_id and 'live:'||source_event_uid=p_payload->>'source_key';
  update public.live_sessions set state=case when state ? 'event' then jsonb_set(state,'{event,round_robin_winner}',display_decision)
    else jsonb_set(state,'{page_state,event,round_robin_winner}',display_decision) end,updated_at=now(),version=version+1
    where club_id=p_club_id and 'live:'||coalesce(state->'event'->>'sourceEventUid',state->'page_state'->'event'->>'sourceEventUid')=p_payload->>'source_key';
  update public.badge_program_state set revision=revision+1,retry_after=now(),pending_ties=(select coalesce(jsonb_agg(t),'[]')
    from jsonb_array_elements(pending_ties) t where t->>'source_key'<>p_payload->>'source_key') where club_id=p_club_id;
  result:=jsonb_build_object('ok',true,'winner_player_id',winner,'source_key',p_payload->>'source_key');
  insert into public.badge_round_robin_operations(club_id,operation_id,actor_user_id,payload,result)
  values(p_club_id,p_operation_id,p_actor_user_id,p_payload,result);
  return result;
end $$;

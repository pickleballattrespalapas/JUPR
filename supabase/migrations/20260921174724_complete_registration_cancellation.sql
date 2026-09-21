-- Cancellation removes the operational registration. Commerce IDs remain
-- immutable historical references, so cancelling never erases payment evidence.
alter table public.tournament_commerce_orders
  drop constraint tournament_commerce_orders_registration_id_fkey;
alter table public.tournament_commerce_order_revisions
  drop constraint tournament_commerce_order_revisions_registration_id_fkey;
alter table public.tournament_commerce_promotion_claims
  drop constraint tournament_commerce_promotion_claims_registration_id_fkey;
comment on column public.tournament_commerce_orders.registration_id is
  'Original registration ID; retained as historical evidence after cancellation removes the registration.';

-- Replace live-registration FKs with insert-time validation. Existing order
-- history remains immutable and still belongs to its original order.
create or replace function private.guard_commerce_registration_reference()
returns trigger language plpgsql security invoker set search_path = '' as $$
begin
  if tg_table_name = 'tournament_commerce_orders' then
    perform 1 from public.tournament_registrations
      where id = new.registration_id and tournament_id = new.tournament_id::text for key share;
    if not found then raise exception 'JUPR_CANCEL_INVALID: Registration no longer exists for this order.'; end if;
  else
    perform 1 from public.tournament_commerce_orders
      where id = new.order_id and registration_id = new.registration_id for key share;
    if not found then raise exception 'JUPR_CANCEL_INVALID: Commerce registration does not match the order.'; end if;
  end if;
  return new;
end;
$$;
revoke all on function private.guard_commerce_registration_reference() from public, anon, authenticated;
grant execute on function private.guard_commerce_registration_reference() to service_role;
create trigger guard_commerce_order_registration before insert or update of registration_id
  on public.tournament_commerce_orders for each row execute function private.guard_commerce_registration_reference();
create trigger guard_commerce_revision_registration before insert
  on public.tournament_commerce_order_revisions for each row execute function private.guard_commerce_registration_reference();
create trigger guard_commerce_claim_registration before insert or update of registration_id
  on public.tournament_commerce_promotion_claims for each row execute function private.guard_commerce_registration_reference();

create table private.tournament_registration_cancellations (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null,
  registration_id text not null,
  actor_email text not null,
  cancelled_at timestamptz not null default now(),
  before_json jsonb not null,
  after_json jsonb not null
);
alter table private.tournament_registration_cancellations enable row level security;
revoke all on private.tournament_registration_cancellations from public, anon, authenticated;
grant select, insert on private.tournament_registration_cancellations to service_role;

create or replace function public.server_cancel_tournament_registrations(
  p_tournament_id text,
  p_changes jsonb,
  p_actor_email text
)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  v_ids text[];
  v_selection_ids text[];
  v_scope text[];
  v_survivors text[];
  v_id text;
  v_change jsonb;
  v_registration public.tournament_registrations%rowtype;
  v_removed jsonb := '[]';
  v_before jsonb;
begin
  if jsonb_typeof(p_changes) is distinct from 'array'
     or jsonb_array_length(p_changes) not between 1 and 100 then
    raise exception 'JUPR_CANCEL_INVALID: Choose between one and 100 registrations.';
  end if;
  select array_agg(distinct item->>'id' order by item->>'id') into v_ids
    from jsonb_array_elements(p_changes) item;
  if cardinality(v_ids) <> jsonb_array_length(p_changes)
     or exists (select 1 from jsonb_array_elements(p_changes) item
                where nullif(item->>'id', '') is null or nullif(item->>'expected_updated_at', '') is null) then
    raise exception 'JUPR_CANCEL_INVALID: Unique registrations and their versions are required.';
  end if;

  -- Use the pairing engine's parent-before-selection lock order. Lock the
  -- tournament graph so invitations, pairing and public edits cannot race us.
  for v_id in select id from public.tournament_registrations
    where tournament_id = p_tournament_id order by id loop
    perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('jupr:tournament-registration:' || v_id, 0));
  end loop;
  select array_agg(id order by id) into v_scope from public.tournament_registration_selections
    where tournament_id = p_tournament_id;
  if coalesce(cardinality(v_scope), 0) > 0 then
    perform private.lock_tournament_registration_selection_scope(v_scope);
  end if;

  for v_change in select value from jsonb_array_elements(p_changes) loop
    select * into v_registration from public.tournament_registrations
      where tournament_id = p_tournament_id and id = v_change->>'id' for update;
    if not found or v_registration.updated_at is distinct from (v_change->>'expected_updated_at')::timestamptz then
      raise exception 'JUPR_CANCEL_CONFLICT: Registration changed. Refresh and try again.';
    end if;
    if exists (select 1 from jsonb_object_keys(coalesce(v_change->'patch', '{}')) k
               where k not in ('notes', 'payment_status')) then
      raise exception 'JUPR_CANCEL_INVALID: Unsupported cancellation field.';
    end if;
    if v_change->'patch' ? 'payment_status' and
       coalesce(v_change->'patch'->>'payment_status', '') not in ('unpaid','paid','waived','refunded') then
      raise exception 'JUPR_CANCEL_INVALID: Invalid payment status.';
    end if;
  end loop;
  select coalesce(array_agg(id), array[]::text[]) into v_selection_ids
    from public.tournament_registration_selections where registration_id = any(v_ids) and tournament_id = p_tournament_id;

  select coalesce(array_agg(s.id), array[]::text[]) into v_survivors
  from public.tournament_registration_selections s
  where s.tournament_id = p_tournament_id and not (s.registration_id = any(v_ids))
    and (
      exists (select 1 from public.tournament_registration_team_links l
        where l.status in ('CONFIRMED','ADMIN_CONFIRMED')
          and ((l.selection1_id = s.id and l.selection2_id = any(v_selection_ids))
            or (l.selection2_id = s.id and l.selection1_id = any(v_selection_ids))))
      or exists (select 1 from public.tournament_registrations r
        join public.tournament_registration_selections gone on gone.registration_id = r.id
        where r.id = any(v_ids) and gone.event_option_id = s.event_option_id
          and nullif(lower(btrim(s.partner_email)), '') = lower(btrim(r.email)))
    )
    -- An old text reference must never break a newer confirmed partnership.
    and not exists (select 1 from public.tournament_registration_team_links l
      where l.status in ('CONFIRMED','ADMIN_CONFIRMED') and s.id in (l.selection1_id,l.selection2_id)
        and not (l.selection1_id = any(v_selection_ids) or l.selection2_id = any(v_selection_ids)));

  if exists (
    select 1 from public.tournament_registration_selections s
    join public.tournament_registrations r on r.id = s.registration_id
    join public.tournament_teams t on t.tournament_id::text = s.tournament_id
      and upper(coalesce(t.source, '')) in ('REGISTRATION','REGISTRATION_COMBINED_RATING')
      and (t.source_selection_id = s.id or r.player_id in (t.player1_id,t.player2_id))
    left join public.tournament_event_draws d on d.id = t.draw_id and d.tournament_id = t.tournament_id
    where (s.id = any(v_selection_ids) or s.id = any(v_survivors))
      and coalesce(d.registration_day_id::text,t.registration_day_id::text) = s.registration_day_id
      and coalesce(d.event_option_id::text,t.event_option_id::text) = s.event_option_id
  ) then
    raise exception 'JUPR_CANCEL_IMPORTED: Remove the affected draw team in Tournament Ops before cancelling this registration.';
  end if;
  if exists (select 1 from public.tournament_four_player_teams where captain_registration_id = any(v_ids)) then
    raise exception 'JUPR_CANCEL_CAPTAIN: Assign a new four-player team captain before cancelling this registration.';
  end if;

  -- Audit, release reserved commerce, sever relationships, and remove entries in
  -- one transaction. A stale row, fulfilled extra or finalized team rolls it all back.
  for v_change in select value from jsonb_array_elements(p_changes) loop
    select * into v_registration from public.tournament_registrations where id = v_change->>'id';
    v_before := jsonb_build_object('registration', to_jsonb(v_registration),
      'selections', (select coalesce(jsonb_agg(to_jsonb(s)), '[]') from public.tournament_registration_selections s where s.registration_id = v_registration.id),
      'team_links', (select coalesce(jsonb_agg(to_jsonb(l)), '[]') from public.tournament_registration_team_links l where v_registration.id in (l.registration1_id,l.registration2_id)));
    update public.tournament_registrations set status = 'cancelled', updated_at = clock_timestamp(),
      notes = case when v_change->'patch' ? 'notes' then v_change->'patch'->>'notes' else notes end,
      payment_status = coalesce(v_change->'patch'->>'payment_status', payment_status)
      where id = v_registration.id returning * into v_registration;
    v_removed := v_removed || jsonb_build_array(to_jsonb(v_registration) || jsonb_build_object('removed', true));
    insert into private.tournament_registration_cancellations(tournament_id,registration_id,actor_email,before_json,after_json)
      values(p_tournament_id::uuid,v_registration.id,coalesce(p_actor_email,''),v_before,
        jsonb_build_object('removed', true, 'registration', to_jsonb(v_registration), 'released_selection_ids', to_jsonb(v_survivors)));
  end loop;

  delete from public.tournament_partner_invitations i where i.tournament_id::text = p_tournament_id
    and (i.target_selection_id = any(v_selection_ids) or i.requester_selection_id = any(v_selection_ids)
      or lower(btrim(i.requester_email)) in (select lower(btrim(email)) from public.tournament_registrations where id = any(v_ids)));
  delete from public.tournament_registration_team_links where tournament_id::text = p_tournament_id
    and (registration1_id = any(v_ids) or registration2_id = any(v_ids)
      or selection1_id = any(v_selection_ids) or selection2_id = any(v_selection_ids));
  delete from public.tournament_registration_partner_requests where tournament_id::text = p_tournament_id
    and (requester_registration_id = any(v_ids) or target_registration_id = any(v_ids)
      or created_by_registration_id = any(v_ids)
      or requester_selection_id = any(v_selection_ids) or target_selection_id = any(v_selection_ids));
  delete from public.tournament_four_player_team_members where tournament_id::text = p_tournament_id and registration_id = any(v_ids);
  delete from public.tournament_rating_eligibility_reviews where tournament_id::text = p_tournament_id and partner_registration_id = any(v_ids);
  update public.tournament_registration_selections set partner_mode = 'NEEDS_PARTNER',
    partner_name = null, partner_email = null, partner_phone = null, partner_dupr_id = null,
    partner_skill = null, partner_age = null, partner_gender = null, show_on_partner_board = true
    where id = any(v_survivors);
  update public.tournament_registrations set updated_at = clock_timestamp()
    where id in (select registration_id from public.tournament_registration_selections where id = any(v_survivors));
  delete from public.tournament_registrations where tournament_id = p_tournament_id and id = any(v_ids);
  return jsonb_build_object('ok', true, 'registrations', v_removed, 'released_selection_ids', to_jsonb(v_survivors));
end;
$$;
revoke all on function public.server_cancel_tournament_registrations(text,jsonb,text) from public, anon, authenticated;
grant execute on function public.server_cancel_tournament_registrations(text,jsonb,text) to service_role;

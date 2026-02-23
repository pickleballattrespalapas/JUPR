begin;

update public.session_ladder_sessions
set state = case lower(state)
  when 'draft' then 'draft'
  when 'roster_open' then 'roster_open'
  when 'seeded_locked' then 'seeded_locked'
  when 'round_1_active' then 'round_1_active'
  when 'round_1_closed' then 'round_1_closed'
  when 'round_2_active' then 'round_2_active'
  when 'round_2_closed' then 'round_2_closed'
  when 'round_3_active' then 'round_3_active'
  when 'round_3_closed' then 'round_3_closed'
  when 'completed' then 'completed'
  when 'published' then 'published'
  when 'setup' then 'seeded_locked'
  when 'active' then 'roster_open'
  when 'cancelled' then 'completed'
  when 'archived' then 'published'
  else lower(state)
end;

alter table if exists public.session_ladder_sessions
  drop constraint if exists session_ladder_sessions_state_check;

alter table if exists public.session_ladder_sessions
  add constraint session_ladder_sessions_state_check
  check (
    state in (
      'draft',
      'roster_open',
      'seeded_locked',
      'round_1_active',
      'round_1_closed',
      'round_2_active',
      'round_2_closed',
      'round_3_active',
      'round_3_closed',
      'completed',
      'published'
    )
  );

commit;

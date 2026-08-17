-- Give the day foreign key a matching leading-column index for delete checks.
-- The existing (tournament_id, registration_day_id, attendance_status) index
-- cannot support lookups that begin with registration_day_id alone.

create index if not exists idx_tournament_registration_check_ins_registration_day
  on public.tournament_registration_check_ins (registration_day_id);

-- Auto-confirm tournament registrations; pending admin confirmation is retired.

alter table if exists tournament_registrations
  alter column status set default 'confirmed';

update tournament_registrations
set status = 'confirmed'
where lower(status) = 'pending';

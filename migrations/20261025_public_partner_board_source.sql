alter table if exists public.tournament_registration_partner_requests
  drop constraint if exists tournament_registration_partner_requests_source_chk;

alter table if exists public.tournament_registration_partner_requests
  add constraint tournament_registration_partner_requests_source_chk
  check (
    source in (
      'PROFILE_SEARCH',
      'NEEDS_PARTNER_LIST',
      'PUBLIC_PARTNER_BOARD',
      'ADMIN_RECONCILIATION',
      'ADMIN_CREATED',
      'INVITE_LINK',
      'LEGACY_TEXT_MATCH'
    )
  );

-- Tournament add-ons, fixed-inventory packages, bundles, promotions, immutable
-- order revisions, fulfillment, and atomic recovery.
--
-- This is an additive, staging-first migration.  Every new table and RPC is
-- server-only: API clients receive no table or function privileges.

do $preflight$
begin
  if to_regclass('public.clubs') is null
     or to_regclass('public.tournaments') is null
     or to_regclass('public.tournament_event_options') is null
     or to_regclass('public.tournament_registrations') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament commerce requires clubs, tournaments, event options, and registrations.';
  end if;
end
$preflight$;

create table if not exists public.tournament_commerce_catalog_state (
  tournament_id uuid primary key references public.tournaments(id) on delete cascade,
  club_id text not null references public.clubs(id) on delete cascade,
  currency text not null default 'USD',
  catalog_revision integer not null default 0,
  catalog_fingerprint text not null default '',
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_catalog_currency_chk check (currency = 'USD'),
  constraint tournament_commerce_catalog_revision_chk check (catalog_revision >= 0)
);

create index if not exists tournament_commerce_catalog_state_club_fk_idx
  on public.tournament_commerce_catalog_state (club_id);

create table if not exists public.tournament_commerce_items (
  id uuid primary key,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  club_id text not null references public.clubs(id) on delete cascade,
  name text not null,
  description text not null default '',
  kind text not null,
  status text not null default 'DRAFT',
  base_price_minor integer not null default 0,
  inventory_limit integer,
  max_per_registration integer not null default 1,
  available_from timestamptz,
  available_until timestamptz,
  requires_fulfillment boolean not null default true,
  fulfillment_instructions text not null default '',
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_item_kind_chk
    check (kind in ('MERCHANDISE', 'LODGING', 'MEAL', 'DRINK_PACK', 'OTHER')),
  constraint tournament_commerce_item_status_chk
    check (status in ('DRAFT', 'ACTIVE', 'ARCHIVED')),
  constraint tournament_commerce_item_price_chk check (base_price_minor >= 0),
  constraint tournament_commerce_item_inventory_chk
    check (inventory_limit is null or inventory_limit >= 0),
  constraint tournament_commerce_item_max_chk check (max_per_registration > 0),
  constraint tournament_commerce_item_window_chk
    check (available_until is null or available_from is null or available_until >= available_from)
);

create index if not exists tournament_commerce_items_tournament_fk_idx
  on public.tournament_commerce_items (tournament_id, status, sort_order);
create index if not exists tournament_commerce_items_club_fk_idx
  on public.tournament_commerce_items (club_id);

create table if not exists public.tournament_commerce_item_variants (
  id uuid primary key,
  item_id uuid not null references public.tournament_commerce_items(id) on delete cascade,
  name text not null,
  sku text not null default '',
  status text not null default 'ACTIVE',
  price_delta_minor integer not null default 0,
  inventory_limit integer,
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_variant_status_chk
    check (status in ('DRAFT', 'ACTIVE', 'ARCHIVED')),
  constraint tournament_commerce_variant_inventory_chk
    check (inventory_limit is null or inventory_limit >= 0)
);

create index if not exists tournament_commerce_variants_item_fk_idx
  on public.tournament_commerce_item_variants (item_id, status, sort_order);

create table if not exists public.tournament_commerce_bundles (
  id uuid primary key,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  club_id text not null references public.clubs(id) on delete cascade,
  name text not null,
  description text not null default '',
  status text not null default 'DRAFT',
  price_minor integer not null,
  max_per_registration integer not null default 1,
  available_from timestamptz,
  available_until timestamptz,
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_bundle_status_chk
    check (status in ('DRAFT', 'ACTIVE', 'ARCHIVED')),
  constraint tournament_commerce_bundle_price_chk check (price_minor >= 0),
  constraint tournament_commerce_bundle_max_chk check (max_per_registration > 0),
  constraint tournament_commerce_bundle_window_chk
    check (available_until is null or available_from is null or available_until >= available_from)
);

create index if not exists tournament_commerce_bundles_tournament_fk_idx
  on public.tournament_commerce_bundles (tournament_id, status, sort_order);
create index if not exists tournament_commerce_bundles_club_fk_idx
  on public.tournament_commerce_bundles (club_id);

create table if not exists public.tournament_commerce_bundle_components (
  id uuid primary key,
  bundle_id uuid not null references public.tournament_commerce_bundles(id) on delete cascade,
  component_type text not null,
  event_option_id text references public.tournament_event_options(id) on delete restrict,
  item_id uuid references public.tournament_commerce_items(id) on delete restrict,
  variant_id uuid references public.tournament_commerce_item_variants(id) on delete restrict,
  quantity integer not null default 1,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_component_type_chk
    check (component_type in ('EVENT_OPTION', 'ITEM_VARIANT')),
  constraint tournament_commerce_component_quantity_chk check (quantity > 0),
  constraint tournament_commerce_component_target_chk check (
    (
      component_type = 'EVENT_OPTION'
      and event_option_id is not null
      and item_id is null
      and variant_id is null
    )
    or (
      component_type = 'ITEM_VARIANT'
      and event_option_id is null
      and item_id is not null
      and variant_id is not null
    )
  )
);

create unique index if not exists tournament_commerce_components_unique_target_idx
  on public.tournament_commerce_bundle_components (
    bundle_id,
    component_type,
    coalesce(event_option_id, ''),
    coalesce(variant_id::text, '')
  );
create index if not exists tournament_commerce_components_bundle_fk_idx
  on public.tournament_commerce_bundle_components (bundle_id);
create index if not exists tournament_commerce_components_event_fk_idx
  on public.tournament_commerce_bundle_components (event_option_id)
  where event_option_id is not null;
create index if not exists tournament_commerce_components_item_fk_idx
  on public.tournament_commerce_bundle_components (item_id)
  where item_id is not null;
create index if not exists tournament_commerce_components_variant_fk_idx
  on public.tournament_commerce_bundle_components (variant_id)
  where variant_id is not null;

create table if not exists public.tournament_commerce_promotions (
  id uuid primary key,
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  club_id text not null references public.clubs(id) on delete cascade,
  name text not null,
  promotion_type text not null,
  target_type text not null,
  item_id uuid references public.tournament_commerce_items(id) on delete restrict,
  variant_id uuid references public.tournament_commerce_item_variants(id) on delete restrict,
  bundle_id uuid references public.tournament_commerce_bundles(id) on delete restrict,
  starts_at timestamptz,
  ends_at timestamptz,
  giveaway_limit integer,
  per_registration_limit integer not null default 1,
  priority integer not null default 100,
  status text not null default 'DRAFT',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_promotion_type_chk
    check (promotion_type in ('DATE_WINDOW_FREE', 'FIRST_N_REGISTRANTS', 'FIRST_N_CLAIMS')),
  constraint tournament_commerce_promotion_target_type_chk
    check (target_type in ('ITEM', 'ITEM_VARIANT', 'BUNDLE')),
  constraint tournament_commerce_promotion_status_chk
    check (status in ('DRAFT', 'ACTIVE', 'ARCHIVED')),
  constraint tournament_commerce_promotion_limit_chk check (
    per_registration_limit > 0
    and (
      (promotion_type = 'DATE_WINDOW_FREE' and giveaway_limit is null)
      or giveaway_limit > 0
    )
  ),
  constraint tournament_commerce_promotion_window_chk
    check (ends_at is null or starts_at is null or ends_at >= starts_at),
  constraint tournament_commerce_promotion_target_chk check (
    (
      target_type = 'ITEM'
      and item_id is not null
      and variant_id is null
      and bundle_id is null
    )
    or (
      target_type = 'ITEM_VARIANT'
      and item_id is not null
      and variant_id is not null
      and bundle_id is null
    )
    or (
      target_type = 'BUNDLE'
      and item_id is null
      and variant_id is null
      and bundle_id is not null
    )
  )
);

create index if not exists tournament_commerce_promotions_tournament_fk_idx
  on public.tournament_commerce_promotions (tournament_id, status, priority);
create index if not exists tournament_commerce_promotions_club_fk_idx
  on public.tournament_commerce_promotions (club_id);
create index if not exists tournament_commerce_promotions_item_fk_idx
  on public.tournament_commerce_promotions (item_id)
  where item_id is not null;
create index if not exists tournament_commerce_promotions_variant_fk_idx
  on public.tournament_commerce_promotions (variant_id)
  where variant_id is not null;
create index if not exists tournament_commerce_promotions_bundle_fk_idx
  on public.tournament_commerce_promotions (bundle_id)
  where bundle_id is not null;

create table if not exists public.tournament_commerce_orders (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  club_id text not null references public.clubs(id) on delete restrict,
  registration_id text not null references public.tournament_registrations(id) on delete restrict,
  status text not null default 'ACTIVE',
  payment_status text not null default 'UNPAID',
  currency text not null default 'USD',
  current_revision integer not null default 0,
  request_fingerprint text not null,
  quote_fingerprint text not null,
  list_subtotal_minor integer not null,
  discount_minor integer not null,
  total_minor integer not null,
  cancelled_at timestamptz,
  cancellation_reason text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint tournament_commerce_order_registration_uq unique (registration_id),
  constraint tournament_commerce_order_status_chk
    check (status in ('ACTIVE', 'CANCELLED')),
  constraint tournament_commerce_order_payment_chk
    check (payment_status in ('UNPAID', 'PAID', 'WAIVED', 'REFUNDED')),
  constraint tournament_commerce_order_currency_chk check (currency = 'USD'),
  constraint tournament_commerce_order_revision_chk check (current_revision > 0),
  constraint tournament_commerce_order_amounts_chk check (
    list_subtotal_minor >= 0
    and discount_minor >= 0
    and total_minor >= 0
    and list_subtotal_minor - discount_minor = total_minor
  )
);

create index if not exists tournament_commerce_orders_tournament_fk_idx
  on public.tournament_commerce_orders (tournament_id, status, created_at);
create index if not exists tournament_commerce_orders_club_fk_idx
  on public.tournament_commerce_orders (club_id, status);
create index if not exists tournament_commerce_orders_registration_fk_idx
  on public.tournament_commerce_orders (registration_id);

create table if not exists public.tournament_commerce_order_revisions (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references public.tournament_commerce_orders(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  registration_id text not null references public.tournament_registrations(id) on delete restrict,
  revision integer not null,
  request_fingerprint text not null,
  quote_fingerprint text not null,
  catalog_fingerprint text not null,
  currency text not null,
  list_subtotal_minor integer not null,
  discount_minor integer not null,
  total_minor integer not null,
  price_snapshot jsonb not null,
  actor_type text not null,
  actor_label text not null,
  source text not null,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_order_revision_uq unique (order_id, revision),
  constraint tournament_commerce_revision_currency_chk check (currency = 'USD'),
  constraint tournament_commerce_revision_actor_chk
    check (actor_type in ('PUBLIC_REGISTRANT', 'ADMIN', 'SYSTEM_RECOVERY')),
  constraint tournament_commerce_revision_amounts_chk check (
    list_subtotal_minor >= 0
    and discount_minor >= 0
    and total_minor >= 0
    and list_subtotal_minor - discount_minor = total_minor
  )
);

create index if not exists tournament_commerce_revisions_order_fk_idx
  on public.tournament_commerce_order_revisions (order_id, revision desc);
create index if not exists tournament_commerce_revisions_tournament_fk_idx
  on public.tournament_commerce_order_revisions (tournament_id, created_at);
create index if not exists tournament_commerce_revisions_registration_fk_idx
  on public.tournament_commerce_order_revisions (registration_id);

create table if not exists public.tournament_commerce_order_lines (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references public.tournament_commerce_orders(id) on delete restrict,
  revision_id uuid not null references public.tournament_commerce_order_revisions(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  line_key text not null,
  line_type text not null,
  event_option_id text references public.tournament_event_options(id) on delete restrict,
  item_id uuid references public.tournament_commerce_items(id) on delete restrict,
  variant_id uuid references public.tournament_commerce_item_variants(id) on delete restrict,
  bundle_id uuid references public.tournament_commerce_bundles(id) on delete restrict,
  promotion_id uuid references public.tournament_commerce_promotions(id) on delete restrict,
  label_snapshot text not null,
  option_snapshot text,
  quantity integer not null,
  list_unit_minor integer not null,
  final_unit_minor integer not null,
  list_total_minor integer not null,
  final_total_minor integer not null,
  requires_fulfillment boolean not null default false,
  line_snapshot jsonb not null,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_order_line_revision_key_uq
    unique (revision_id, line_key),
  constraint tournament_commerce_order_line_type_chk
    check (line_type in ('EVENT', 'ITEM', 'BUNDLE')),
  constraint tournament_commerce_order_line_amounts_chk check (
    quantity > 0
    and list_unit_minor >= 0
    and final_unit_minor >= 0
    and final_unit_minor <= list_unit_minor
    and list_total_minor = list_unit_minor * quantity
    and final_total_minor = final_unit_minor * quantity
  )
);

create index if not exists tournament_commerce_lines_order_fk_idx
  on public.tournament_commerce_order_lines (order_id, active);
create index if not exists tournament_commerce_lines_revision_fk_idx
  on public.tournament_commerce_order_lines (revision_id);
create index if not exists tournament_commerce_lines_tournament_fk_idx
  on public.tournament_commerce_order_lines (tournament_id);
create index if not exists tournament_commerce_lines_event_fk_idx
  on public.tournament_commerce_order_lines (event_option_id)
  where event_option_id is not null;
create index if not exists tournament_commerce_lines_item_fk_idx
  on public.tournament_commerce_order_lines (item_id)
  where item_id is not null;
create index if not exists tournament_commerce_lines_variant_fk_idx
  on public.tournament_commerce_order_lines (variant_id)
  where variant_id is not null;
create index if not exists tournament_commerce_lines_bundle_fk_idx
  on public.tournament_commerce_order_lines (bundle_id)
  where bundle_id is not null;
create index if not exists tournament_commerce_lines_promotion_fk_idx
  on public.tournament_commerce_order_lines (promotion_id)
  where promotion_id is not null;

create table if not exists public.tournament_commerce_inventory_reservations (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references public.tournament_commerce_orders(id) on delete restrict,
  revision_id uuid not null references public.tournament_commerce_order_revisions(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  item_id uuid not null references public.tournament_commerce_items(id) on delete restrict,
  variant_id uuid not null references public.tournament_commerce_item_variants(id) on delete restrict,
  quantity integer not null,
  status text not null default 'ACTIVE',
  released_at timestamptz,
  release_reason text,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_inventory_revision_variant_uq
    unique (revision_id, variant_id),
  constraint tournament_commerce_inventory_quantity_chk check (quantity > 0),
  constraint tournament_commerce_inventory_status_chk
    check (status in ('ACTIVE', 'RELEASED'))
);

create index if not exists tournament_commerce_inventory_order_fk_idx
  on public.tournament_commerce_inventory_reservations (order_id, status);
create index if not exists tournament_commerce_inventory_revision_fk_idx
  on public.tournament_commerce_inventory_reservations (revision_id);
create index if not exists tournament_commerce_inventory_tournament_fk_idx
  on public.tournament_commerce_inventory_reservations (
    tournament_id, status, id
  );
create index if not exists tournament_commerce_inventory_item_fk_idx
  on public.tournament_commerce_inventory_reservations (item_id);
create index if not exists tournament_commerce_inventory_variant_fk_idx
  on public.tournament_commerce_inventory_reservations (variant_id, status);

create table if not exists public.tournament_commerce_promotion_claims (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references public.tournament_commerce_orders(id) on delete restrict,
  revision_id uuid not null references public.tournament_commerce_order_revisions(id) on delete restrict,
  registration_id text not null references public.tournament_registrations(id) on delete restrict,
  promotion_id uuid not null references public.tournament_commerce_promotions(id) on delete restrict,
  quantity integer not null,
  status text not null default 'ACTIVE',
  released_at timestamptz,
  release_reason text,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_claim_revision_promotion_uq
    unique (revision_id, promotion_id),
  constraint tournament_commerce_claim_quantity_chk check (quantity > 0),
  constraint tournament_commerce_claim_status_chk
    check (status in ('ACTIVE', 'RELEASED'))
);

create index if not exists tournament_commerce_claims_order_fk_idx
  on public.tournament_commerce_promotion_claims (order_id, status);
create index if not exists tournament_commerce_claims_revision_fk_idx
  on public.tournament_commerce_promotion_claims (revision_id);
create index if not exists tournament_commerce_claims_registration_fk_idx
  on public.tournament_commerce_promotion_claims (registration_id);
create index if not exists tournament_commerce_claims_promotion_fk_idx
  on public.tournament_commerce_promotion_claims (promotion_id, status, id);

create index if not exists tournament_registrations_tournament_commerce_page_idx
  on public.tournament_registrations (tournament_id, id);

create table if not exists public.tournament_commerce_fulfillment (
  id uuid primary key default gen_random_uuid(),
  order_id uuid not null references public.tournament_commerce_orders(id) on delete restrict,
  revision_id uuid not null references public.tournament_commerce_order_revisions(id) on delete restrict,
  order_line_id uuid not null references public.tournament_commerce_order_lines(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  item_id uuid not null references public.tournament_commerce_items(id) on delete restrict,
  variant_id uuid not null references public.tournament_commerce_item_variants(id) on delete restrict,
  label_snapshot text not null,
  option_snapshot text,
  sku_snapshot text not null default '',
  instructions_snapshot text not null default '',
  status text not null default 'PENDING',
  quantity integer not null,
  notes text not null default '',
  fulfilled_at timestamptz,
  fulfilled_by text,
  updated_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  constraint tournament_commerce_fulfillment_line_variant_uq
    unique (order_line_id, variant_id),
  constraint tournament_commerce_fulfillment_status_chk
    check (status in ('PENDING', 'READY', 'FULFILLED', 'CANCELLED')),
  constraint tournament_commerce_fulfillment_quantity_chk check (quantity > 0)
);

create index if not exists tournament_commerce_fulfillment_order_fk_idx
  on public.tournament_commerce_fulfillment (order_id, status);
create index if not exists tournament_commerce_fulfillment_revision_fk_idx
  on public.tournament_commerce_fulfillment (revision_id);
create index if not exists tournament_commerce_fulfillment_line_fk_idx
  on public.tournament_commerce_fulfillment (order_line_id);
create index if not exists tournament_commerce_fulfillment_tournament_fk_idx
  on public.tournament_commerce_fulfillment (tournament_id, status);
create index if not exists tournament_commerce_fulfillment_item_fk_idx
  on public.tournament_commerce_fulfillment (item_id);
create index if not exists tournament_commerce_fulfillment_variant_fk_idx
  on public.tournament_commerce_fulfillment (variant_id);

create table if not exists public.tournament_commerce_operations (
  id uuid primary key default gen_random_uuid(),
  club_id text not null references public.clubs(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  order_id uuid references public.tournament_commerce_orders(id) on delete restrict,
  idempotency_key uuid not null,
  request_fingerprint text not null,
  action text not null,
  status text not null default 'INTENT',
  actor_type text not null,
  actor_label text not null,
  source text not null,
  attempt_count integer not null default 1,
  result_json jsonb,
  error_text text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz,
  constraint tournament_commerce_operation_idempotency_uq
    unique (club_id, tournament_id, idempotency_key),
  constraint tournament_commerce_operation_status_chk
    check (status in ('INTENT', 'COMPLETED', 'RECOVERY_REQUIRED')),
  constraint tournament_commerce_operation_actor_chk
    check (actor_type in ('PUBLIC_REGISTRANT', 'ADMIN', 'SYSTEM_RECOVERY')),
  constraint tournament_commerce_operation_attempt_chk check (attempt_count > 0)
);

create index if not exists tournament_commerce_operations_club_fk_idx
  on public.tournament_commerce_operations (club_id, created_at desc);
create index if not exists tournament_commerce_operations_tournament_fk_idx
  on public.tournament_commerce_operations (tournament_id, created_at desc);
create index if not exists tournament_commerce_operations_order_fk_idx
  on public.tournament_commerce_operations (order_id, created_at desc)
  where order_id is not null;

create table if not exists public.tournament_commerce_audit_log (
  id bigint generated always as identity primary key,
  club_id text not null references public.clubs(id) on delete restrict,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  order_id uuid references public.tournament_commerce_orders(id) on delete restrict,
  operation_id uuid references public.tournament_commerce_operations(id) on delete restrict,
  actor_type text not null,
  actor_label text not null,
  action text not null,
  before_json jsonb,
  after_json jsonb,
  source text not null,
  created_at timestamptz not null default now(),
  constraint tournament_commerce_audit_actor_chk
    check (actor_type in ('PUBLIC_REGISTRANT', 'ADMIN', 'SYSTEM_RECOVERY'))
);

create index if not exists tournament_commerce_audit_club_fk_idx
  on public.tournament_commerce_audit_log (club_id, created_at desc);
create index if not exists tournament_commerce_audit_tournament_fk_idx
  on public.tournament_commerce_audit_log (tournament_id, created_at desc);
create index if not exists tournament_commerce_audit_order_fk_idx
  on public.tournament_commerce_audit_log (order_id, created_at desc)
  where order_id is not null;
create index if not exists tournament_commerce_audit_operation_fk_idx
  on public.tournament_commerce_audit_log (operation_id)
  where operation_id is not null;

-- No browser role may read or mutate commerce data directly.
alter table public.tournament_commerce_catalog_state enable row level security;
alter table public.tournament_commerce_catalog_state force row level security;
alter table public.tournament_commerce_items enable row level security;
alter table public.tournament_commerce_items force row level security;
alter table public.tournament_commerce_item_variants enable row level security;
alter table public.tournament_commerce_item_variants force row level security;
alter table public.tournament_commerce_bundles enable row level security;
alter table public.tournament_commerce_bundles force row level security;
alter table public.tournament_commerce_bundle_components enable row level security;
alter table public.tournament_commerce_bundle_components force row level security;
alter table public.tournament_commerce_promotions enable row level security;
alter table public.tournament_commerce_promotions force row level security;
alter table public.tournament_commerce_orders enable row level security;
alter table public.tournament_commerce_orders force row level security;
alter table public.tournament_commerce_order_revisions enable row level security;
alter table public.tournament_commerce_order_revisions force row level security;
alter table public.tournament_commerce_order_lines enable row level security;
alter table public.tournament_commerce_order_lines force row level security;
alter table public.tournament_commerce_inventory_reservations enable row level security;
alter table public.tournament_commerce_inventory_reservations force row level security;
alter table public.tournament_commerce_promotion_claims enable row level security;
alter table public.tournament_commerce_promotion_claims force row level security;
alter table public.tournament_commerce_fulfillment enable row level security;
alter table public.tournament_commerce_fulfillment force row level security;
alter table public.tournament_commerce_operations enable row level security;
alter table public.tournament_commerce_operations force row level security;
alter table public.tournament_commerce_audit_log enable row level security;
alter table public.tournament_commerce_audit_log force row level security;

revoke all on table public.tournament_commerce_catalog_state from public, anon, authenticated;
revoke all on table public.tournament_commerce_items from public, anon, authenticated;
revoke all on table public.tournament_commerce_item_variants from public, anon, authenticated;
revoke all on table public.tournament_commerce_bundles from public, anon, authenticated;
revoke all on table public.tournament_commerce_bundle_components from public, anon, authenticated;
revoke all on table public.tournament_commerce_promotions from public, anon, authenticated;
revoke all on table public.tournament_commerce_orders from public, anon, authenticated;
revoke all on table public.tournament_commerce_order_revisions from public, anon, authenticated;
revoke all on table public.tournament_commerce_order_lines from public, anon, authenticated;
revoke all on table public.tournament_commerce_inventory_reservations from public, anon, authenticated;
revoke all on table public.tournament_commerce_promotion_claims from public, anon, authenticated;
revoke all on table public.tournament_commerce_fulfillment from public, anon, authenticated;
revoke all on table public.tournament_commerce_operations from public, anon, authenticated;
revoke all on table public.tournament_commerce_audit_log from public, anon, authenticated;

grant select, insert, update, delete on table public.tournament_commerce_catalog_state to service_role;
grant select, insert, update, delete on table public.tournament_commerce_items to service_role;
grant select, insert, update, delete on table public.tournament_commerce_item_variants to service_role;
grant select, insert, update, delete on table public.tournament_commerce_bundles to service_role;
grant select, insert, update, delete on table public.tournament_commerce_bundle_components to service_role;
grant select, insert, update, delete on table public.tournament_commerce_promotions to service_role;
grant select, insert, update on table public.tournament_commerce_orders to service_role;
grant select, insert on table public.tournament_commerce_order_revisions to service_role;
grant select, insert, update on table public.tournament_commerce_order_lines to service_role;
grant select, insert, update on table public.tournament_commerce_inventory_reservations to service_role;
grant select, insert, update on table public.tournament_commerce_promotion_claims to service_role;
grant select, insert, update on table public.tournament_commerce_fulfillment to service_role;
grant select, insert, update on table public.tournament_commerce_operations to service_role;
grant select, insert on table public.tournament_commerce_audit_log to service_role;
grant usage, select on sequence public.tournament_commerce_audit_log_id_seq to service_role;

create or replace function private.tournament_commerce_reject_immutable_change()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  raise exception using
    errcode = '55000',
    message = 'JUPR_TOURNAMENT_COMMERCE_IMMUTABLE: order revisions and audit rows cannot be changed.';
end
$function$;

revoke all on function private.tournament_commerce_reject_immutable_change()
  from public, anon, authenticated;

drop trigger if exists tournament_commerce_revision_immutable
  on public.tournament_commerce_order_revisions;
create trigger tournament_commerce_revision_immutable
before update or delete on public.tournament_commerce_order_revisions
for each row execute function private.tournament_commerce_reject_immutable_change();

drop trigger if exists tournament_commerce_audit_immutable
  on public.tournament_commerce_audit_log;
create trigger tournament_commerce_audit_immutable
before update or delete on public.tournament_commerce_audit_log
for each row execute function private.tournament_commerce_reject_immutable_change();

create or replace function private.tournament_commerce_operation_begin(
  p_club_id text,
  p_tournament_id uuid,
  p_idempotency_key uuid,
  p_request_fingerprint text,
  p_action text,
  p_actor_type text,
  p_actor_label text,
  p_source text
)
returns public.tournament_commerce_operations
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.tournament_commerce_operations%rowtype;
begin
  insert into public.tournament_commerce_operations (
    club_id,
    tournament_id,
    idempotency_key,
    request_fingerprint,
    action,
    status,
    actor_type,
    actor_label,
    source
  )
  values (
    p_club_id,
    p_tournament_id,
    p_idempotency_key,
    p_request_fingerprint,
    p_action,
    'INTENT',
    p_actor_type,
    p_actor_label,
    p_source
  )
  on conflict (club_id, tournament_id, idempotency_key) do nothing
  returning * into v_operation;

  if found then
    return v_operation;
  end if;

  -- A concurrent request may be inserting the same key. ON CONFLICT waits for
  -- that transaction, and this fresh statement then locks the committed row
  -- before replay validation.
  select operation.*
  into v_operation
  from public.tournament_commerce_operations operation
  where operation.club_id = p_club_id
    and operation.tournament_id = p_tournament_id
    and operation.idempotency_key = p_idempotency_key
  for update;

  if not found then
    raise exception using
      errcode = '55000',
      message = 'JUPR_TOURNAMENT_COMMERCE_OPERATION_RECOVERY_REQUIRED';
  end if;
  if v_operation.request_fingerprint <> p_request_fingerprint
     or v_operation.action <> p_action then
    raise exception using
      errcode = '23505',
      message = 'JUPR_TOURNAMENT_COMMERCE_IDEMPOTENCY_CONFLICT';
  end if;
  update public.tournament_commerce_operations
  set attempt_count = attempt_count + 1,
      updated_at = now()
  where id = v_operation.id
  returning * into v_operation;
  return v_operation;
end
$function$;

revoke all on function private.tournament_commerce_operation_begin(
  text, uuid, uuid, text, text, text, text, text
) from public, anon, authenticated;
grant execute on function private.tournament_commerce_operation_begin(
  text, uuid, uuid, text, text, text, text, text
) to service_role;

create or replace function public.server_replace_tournament_commerce_catalog(
  p_club_id text,
  p_tournament_id uuid,
  p_expected_catalog_fingerprint text,
  p_catalog jsonb,
  p_idempotency_key uuid,
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
  v_tournament public.tournaments%rowtype;
  v_state public.tournament_commerce_catalog_state%rowtype;
  v_operation public.tournament_commerce_operations%rowtype;
  v_result jsonb;
  v_new_fingerprint text;
  v_currency text;
begin
  select tournament.*
  into v_tournament
  from public.tournaments tournament
  where tournament.id = p_tournament_id
    and tournament.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_TOURNAMENT_NOT_FOUND';
  end if;

  if p_catalog is null or jsonb_typeof(p_catalog) <> 'object' then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_INVALID_CATALOG';
  end if;
  v_currency := coalesce(nullif(p_catalog ->> 'currency', ''), 'USD');
  v_new_fingerprint := nullif(p_catalog ->> 'catalog_fingerprint', '');
  if v_currency <> 'USD' or v_new_fingerprint is null then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_INVALID_CATALOG';
  end if;

  v_operation := private.tournament_commerce_operation_begin(
    p_club_id,
    p_tournament_id,
    p_idempotency_key,
    p_request_fingerprint,
    'CATALOG_REPLACE',
    'ADMIN',
    p_actor_label,
    p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;

  insert into public.tournament_commerce_catalog_state (
    tournament_id, club_id, currency, catalog_revision, catalog_fingerprint
  )
  values (p_tournament_id, p_club_id, 'USD', 0, '')
  on conflict (tournament_id) do nothing;

  select state.*
  into v_state
  from public.tournament_commerce_catalog_state state
  where state.tournament_id = p_tournament_id
  for update;
  if v_state.club_id <> p_club_id then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_CLUB_MISMATCH';
  end if;
  if v_state.catalog_fingerprint <> coalesce(p_expected_catalog_fingerprint, '') then
    raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_CATALOG_STALE';
  end if;

  -- Archive removed rows rather than deleting historical catalog identities.
  update public.tournament_commerce_promotions
  set status = 'ARCHIVED', updated_at = now()
  where tournament_id = p_tournament_id
    and not exists (
      select 1
      from jsonb_array_elements(coalesce(p_catalog -> 'promotions', '[]'::jsonb)) row
      where (row ->> 'id')::uuid = tournament_commerce_promotions.id
    );
  update public.tournament_commerce_bundles
  set status = 'ARCHIVED', updated_at = now()
  where tournament_id = p_tournament_id
    and not exists (
      select 1
      from jsonb_array_elements(coalesce(p_catalog -> 'bundles', '[]'::jsonb)) row
      where (row ->> 'id')::uuid = tournament_commerce_bundles.id
    );
  update public.tournament_commerce_item_variants variant
  set status = 'ARCHIVED', updated_at = now()
  where exists (
      select 1 from public.tournament_commerce_items item
      where item.id = variant.item_id and item.tournament_id = p_tournament_id
    )
    and not exists (
      select 1
      from jsonb_array_elements(coalesce(p_catalog -> 'variants', '[]'::jsonb)) row
      where (row ->> 'id')::uuid = variant.id
    );
  update public.tournament_commerce_items
  set status = 'ARCHIVED', updated_at = now()
  where tournament_id = p_tournament_id
    and not exists (
      select 1
      from jsonb_array_elements(coalesce(p_catalog -> 'items', '[]'::jsonb)) row
      where (row ->> 'id')::uuid = tournament_commerce_items.id
    );

  insert into public.tournament_commerce_items (
    id, tournament_id, club_id, name, description, kind, status,
    base_price_minor, inventory_limit, max_per_registration,
    available_from, available_until, requires_fulfillment,
    fulfillment_instructions, sort_order, updated_at
  )
  select
    row.id, p_tournament_id, p_club_id, row.name, coalesce(row.description, ''),
    row.kind, row.status, row.base_price_minor, row.inventory_limit,
    row.max_per_registration, row.available_from, row.available_until,
    row.requires_fulfillment, coalesce(row.fulfillment_instructions, ''),
    row.sort_order, now()
  from jsonb_to_recordset(coalesce(p_catalog -> 'items', '[]'::jsonb)) as row(
    id uuid, name text, description text, kind text, status text,
    base_price_minor integer, inventory_limit integer,
    max_per_registration integer, available_from timestamptz,
    available_until timestamptz, requires_fulfillment boolean,
    fulfillment_instructions text, sort_order integer
  )
  on conflict (id) do update set
    name = excluded.name,
    description = excluded.description,
    kind = excluded.kind,
    status = excluded.status,
    base_price_minor = excluded.base_price_minor,
    inventory_limit = excluded.inventory_limit,
    max_per_registration = excluded.max_per_registration,
    available_from = excluded.available_from,
    available_until = excluded.available_until,
    requires_fulfillment = excluded.requires_fulfillment,
    fulfillment_instructions = excluded.fulfillment_instructions,
    sort_order = excluded.sort_order,
    updated_at = now()
  where tournament_commerce_items.tournament_id = p_tournament_id
    and tournament_commerce_items.club_id = p_club_id;

  insert into public.tournament_commerce_item_variants (
    id, item_id, name, sku, status, price_delta_minor,
    inventory_limit, sort_order, updated_at
  )
  select
    row.id, row.item_id, row.name, coalesce(row.sku, ''), row.status,
    row.price_delta_minor, row.inventory_limit, row.sort_order, now()
  from jsonb_to_recordset(coalesce(p_catalog -> 'variants', '[]'::jsonb)) as row(
    id uuid, item_id uuid, name text, sku text, status text,
    price_delta_minor integer, inventory_limit integer, sort_order integer
  )
  join public.tournament_commerce_items item
    on item.id = row.item_id
   and item.tournament_id = p_tournament_id
   and item.club_id = p_club_id
  on conflict (id) do update set
    item_id = excluded.item_id,
    name = excluded.name,
    sku = excluded.sku,
    status = excluded.status,
    price_delta_minor = excluded.price_delta_minor,
    inventory_limit = excluded.inventory_limit,
    sort_order = excluded.sort_order,
    updated_at = now()
  where exists (
    select 1
    from public.tournament_commerce_items current_item
    where current_item.id = tournament_commerce_item_variants.item_id
      and current_item.tournament_id = p_tournament_id
      and current_item.club_id = p_club_id
  );

  insert into public.tournament_commerce_bundles (
    id, tournament_id, club_id, name, description, status, price_minor,
    max_per_registration, available_from, available_until, sort_order, updated_at
  )
  select
    row.id, p_tournament_id, p_club_id, row.name, coalesce(row.description, ''),
    row.status, row.price_minor, row.max_per_registration,
    row.available_from, row.available_until, row.sort_order, now()
  from jsonb_to_recordset(coalesce(p_catalog -> 'bundles', '[]'::jsonb)) as row(
    id uuid, name text, description text, status text, price_minor integer,
    max_per_registration integer, available_from timestamptz,
    available_until timestamptz, sort_order integer
  )
  on conflict (id) do update set
    name = excluded.name,
    description = excluded.description,
    status = excluded.status,
    price_minor = excluded.price_minor,
    max_per_registration = excluded.max_per_registration,
    available_from = excluded.available_from,
    available_until = excluded.available_until,
    sort_order = excluded.sort_order,
    updated_at = now()
  where tournament_commerce_bundles.tournament_id = p_tournament_id
    and tournament_commerce_bundles.club_id = p_club_id;

  delete from public.tournament_commerce_bundle_components component
  using public.tournament_commerce_bundles bundle
  where component.bundle_id = bundle.id
    and bundle.tournament_id = p_tournament_id;

  insert into public.tournament_commerce_bundle_components (
    id, bundle_id, component_type, event_option_id, item_id, variant_id, quantity
  )
  select
    row.id, row.bundle_id, row.component_type, row.event_option_id,
    row.item_id, row.variant_id, row.quantity
  from jsonb_to_recordset(
    coalesce(p_catalog -> 'bundle_components', '[]'::jsonb)
  ) as row(
    id uuid, bundle_id uuid, component_type text, event_option_id text,
    item_id uuid, variant_id uuid, quantity integer
  )
  join public.tournament_commerce_bundles bundle
    on bundle.id = row.bundle_id
   and bundle.tournament_id = p_tournament_id
   and bundle.club_id = p_club_id;

  insert into public.tournament_commerce_promotions (
    id, tournament_id, club_id, name, promotion_type, target_type,
    item_id, variant_id, bundle_id, starts_at, ends_at, giveaway_limit,
    per_registration_limit, priority, status, updated_at
  )
  select
    row.id, p_tournament_id, p_club_id, row.name, row.promotion_type,
    row.target_type, row.item_id, row.variant_id, row.bundle_id,
    row.starts_at, row.ends_at, row.giveaway_limit,
    row.per_registration_limit, row.priority, row.status, now()
  from jsonb_to_recordset(coalesce(p_catalog -> 'promotions', '[]'::jsonb)) as row(
    id uuid, name text, promotion_type text, target_type text, item_id uuid,
    variant_id uuid, bundle_id uuid, starts_at timestamptz, ends_at timestamptz,
    giveaway_limit integer, per_registration_limit integer,
    priority integer, status text
  )
  on conflict (id) do update set
    name = excluded.name,
    promotion_type = excluded.promotion_type,
    target_type = excluded.target_type,
    item_id = excluded.item_id,
    variant_id = excluded.variant_id,
    bundle_id = excluded.bundle_id,
    starts_at = excluded.starts_at,
    ends_at = excluded.ends_at,
    giveaway_limit = excluded.giveaway_limit,
    per_registration_limit = excluded.per_registration_limit,
    priority = excluded.priority,
    status = excluded.status,
    updated_at = now()
  where tournament_commerce_promotions.tournament_id = p_tournament_id
    and tournament_commerce_promotions.club_id = p_club_id;

  update public.tournament_commerce_catalog_state
  set currency = 'USD',
      catalog_revision = catalog_revision + 1,
      catalog_fingerprint = v_new_fingerprint,
      updated_at = now()
  where tournament_id = p_tournament_id
  returning * into v_state;

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'tournament_commerce_catalog_replace',
    'tournament_id', p_tournament_id,
    'catalog_revision', v_state.catalog_revision,
    'catalog_fingerprint', v_state.catalog_fingerprint,
    'idempotent_replay', false,
    'operation_id', v_operation.id
  );
  update public.tournament_commerce_operations
  set status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    p_club_id, p_tournament_id, v_operation.id, 'ADMIN', p_actor_label,
    'CATALOG_REPLACE',
    jsonb_build_object(
      'catalog_revision', v_state.catalog_revision - 1,
      'catalog_fingerprint', p_expected_catalog_fingerprint
    ),
    v_result,
    p_source
  );
  return v_result;
end
$function$;

create or replace function public.server_apply_tournament_commerce_order(
  p_club_id text,
  p_tournament_id uuid,
  p_registration_id text,
  p_expected_order_updated_at timestamptz,
  p_quote_snapshot jsonb,
  p_idempotency_key uuid,
  p_request_fingerprint text,
  p_actor_type text,
  p_actor_label text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_tournament public.tournaments%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_state public.tournament_commerce_catalog_state%rowtype;
  v_order public.tournament_commerce_orders%rowtype;
  v_before jsonb;
  v_revision integer;
  v_revision_id uuid;
  v_operation public.tournament_commerce_operations%rowtype;
  v_result jsonb;
  v_inventory jsonb;
  v_line jsonb;
  v_component jsonb;
  v_promo jsonb;
  v_item public.tournament_commerce_items%rowtype;
  v_variant public.tournament_commerce_item_variants%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_bundle public.tournament_commerce_bundles%rowtype;
  v_promotion public.tournament_commerce_promotions%rowtype;
  v_reserved integer;
  v_item_reserved integer;
  v_claimed integer;
  v_rank integer;
  v_total integer;
  v_list_total integer;
  v_discount integer;
  v_line_id uuid;
  v_pair record;
begin
  select tournament.* into v_tournament
  from public.tournaments tournament
  where tournament.id = p_tournament_id and tournament.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_TOURNAMENT_NOT_FOUND';
  end if;

  select registration.* into v_registration
  from public.tournament_registrations registration
  where registration.id = p_registration_id
    and registration.tournament_id = p_tournament_id::text
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_NOT_FOUND';
  end if;
  if lower(coalesce(v_registration.status, 'confirmed')) = 'cancelled' then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_CANCELLED';
  end if;

  if p_quote_snapshot is null
     or jsonb_typeof(p_quote_snapshot) <> 'object'
     or jsonb_typeof(p_quote_snapshot -> 'lines') <> 'array'
     or jsonb_typeof(p_quote_snapshot -> 'inventory') <> 'array'
     or jsonb_typeof(p_quote_snapshot -> 'applied_promotions') <> 'array'
     or p_quote_snapshot ->> 'currency' <> 'USD'
     or p_quote_snapshot ->> 'request_fingerprint' <> p_request_fingerprint then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_INVALID_QUOTE';
  end if;

  select state.* into v_state
  from public.tournament_commerce_catalog_state state
  where state.tournament_id = p_tournament_id
  for share;
  if not found or v_state.club_id <> p_club_id
     or v_state.catalog_fingerprint <> p_quote_snapshot ->> 'catalog_fingerprint' then
    raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_QUOTE_STALE';
  end if;

  select
    coalesce(sum((line ->> 'final_total_minor')::integer), 0),
    coalesce(sum((line ->> 'list_total_minor')::integer), 0)
  into v_total, v_list_total
  from jsonb_array_elements(p_quote_snapshot -> 'lines') line;
  v_discount := v_list_total - v_total;
  if v_total <> (p_quote_snapshot ->> 'total_minor')::integer
     or v_list_total <> (p_quote_snapshot ->> 'list_subtotal_minor')::integer
     or v_discount <> (p_quote_snapshot ->> 'discount_minor')::integer
     or exists (
       select 1 from jsonb_array_elements(p_quote_snapshot -> 'lines') line
       where (line ->> 'quantity')::integer <= 0
          or (line ->> 'list_unit_minor')::integer < 0
          or (line ->> 'final_unit_minor')::integer < 0
          or (line ->> 'final_unit_minor')::integer > (line ->> 'list_unit_minor')::integer
          or (line ->> 'list_total_minor')::integer
             <> (line ->> 'list_unit_minor')::integer * (line ->> 'quantity')::integer
          or (line ->> 'final_total_minor')::integer
             <> (line ->> 'final_unit_minor')::integer * (line ->> 'quantity')::integer
     ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_QUOTE_TOTAL_MISMATCH';
  end if;

  -- Recheck every list price and promotion target from server-owned catalog
  -- rows.  The API computes the best-price allocation, but client JSON never
  -- becomes pricing authority.
  for v_line in
    select value from jsonb_array_elements(p_quote_snapshot -> 'lines')
  loop
    if v_line ->> 'line_type' = 'EVENT' then
      select event.* into v_event
      from public.tournament_event_options event
      where event.id = v_line ->> 'event_option_id'
        and event.tournament_id = p_tournament_id::text
        and coalesce(event.enabled, true);
      if not found
         or round(coalesce(v_event.price_usd, 0) * 100)::integer
            <> (v_line ->> 'list_unit_minor')::integer
         or nullif(v_line ->> 'promotion_id', '') is not null then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_EVENT_PRICE_INVALID';
      end if;
    elsif v_line ->> 'line_type' = 'ITEM' then
      select variant as variant_row, item as item_row
      into v_pair
      from public.tournament_commerce_item_variants variant
      join public.tournament_commerce_items item on item.id = variant.item_id
      where variant.id = (v_line ->> 'variant_id')::uuid
        and item.id = (v_line ->> 'item_id')::uuid
        and item.tournament_id = p_tournament_id
        and item.club_id = p_club_id
        and item.status = 'ACTIVE'
        and variant.status = 'ACTIVE';
      if not found then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_ITEM_PRICE_INVALID';
      end if;
      v_variant := v_pair.variant_row;
      v_item := v_pair.item_row;
      if v_item.base_price_minor + v_variant.price_delta_minor
            <> (v_line ->> 'list_unit_minor')::integer then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_ITEM_PRICE_INVALID';
      end if;
    elsif v_line ->> 'line_type' = 'BUNDLE' then
      select bundle.* into v_bundle
      from public.tournament_commerce_bundles bundle
      where bundle.id = (v_line ->> 'bundle_id')::uuid
        and bundle.tournament_id = p_tournament_id
        and bundle.club_id = p_club_id
        and bundle.status = 'ACTIVE';
      if not found
         or (
           nullif(v_line ->> 'promotion_id', '') is null
           and v_bundle.price_minor <> (v_line ->> 'final_unit_minor')::integer
         ) then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_BUNDLE_PRICE_INVALID';
      end if;
    else
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_LINE_TYPE_INVALID';
    end if;

    if nullif(v_line ->> 'promotion_id', '') is null then
      if v_line ->> 'line_type' <> 'BUNDLE'
         and (v_line ->> 'final_unit_minor')::integer
             <> (v_line ->> 'list_unit_minor')::integer then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_UNEXPLAINED_DISCOUNT';
      end if;
    else
      select promotion.* into v_promotion
      from public.tournament_commerce_promotions promotion
      where promotion.id = (v_line ->> 'promotion_id')::uuid
        and promotion.tournament_id = p_tournament_id
        and promotion.club_id = p_club_id
        and promotion.status = 'ACTIVE';
      if not found
         or (v_line ->> 'final_unit_minor')::integer <> 0
         or not (
           (
             v_promotion.target_type = 'ITEM'
             and v_line ->> 'line_type' = 'ITEM'
             and v_promotion.item_id = (v_line ->> 'item_id')::uuid
           )
           or (
             v_promotion.target_type = 'ITEM_VARIANT'
             and v_line ->> 'line_type' = 'ITEM'
             and v_promotion.variant_id = (v_line ->> 'variant_id')::uuid
           )
           or (
             v_promotion.target_type = 'BUNDLE'
             and v_line ->> 'line_type' = 'BUNDLE'
             and v_promotion.bundle_id = (v_line ->> 'bundle_id')::uuid
           )
         ) then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_PROMOTION_TARGET_INVALID';
      end if;
    end if;
  end loop;

  v_operation := private.tournament_commerce_operation_begin(
    p_club_id,
    p_tournament_id,
    p_idempotency_key,
    p_request_fingerprint,
    'ORDER_APPLY',
    p_actor_type,
    p_actor_label,
    p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;

  select orders.* into v_order
  from public.tournament_commerce_orders orders
  where orders.registration_id = p_registration_id
  for update;
  if found then
    v_before := to_jsonb(v_order);
    if p_expected_order_updated_at is null
       or v_order.updated_at is distinct from p_expected_order_updated_at then
      raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_STALE';
    end if;
    if v_order.status = 'ACTIVE'
       and v_order.quote_fingerprint = p_quote_snapshot ->> 'quote_fingerprint' then
      v_revision := v_order.current_revision;
      select revision.id into v_revision_id
      from public.tournament_commerce_order_revisions revision
      where revision.order_id = v_order.id
        and revision.revision = v_order.current_revision;
      if not found then
        raise exception using
          errcode = '55000',
          message = 'JUPR_TOURNAMENT_COMMERCE_CURRENT_REVISION_MISSING';
      end if;

      v_result := jsonb_build_object(
        'ok', true,
        'mode', 'tournament_commerce_order_apply',
        'order_id', v_order.id,
        'revision_id', v_revision_id,
        'revision', v_revision,
        'registration_id', p_registration_id,
        'quote_fingerprint', v_order.quote_fingerprint,
        'total_minor', v_order.total_minor,
        'payment_status', v_order.payment_status,
        'idempotent_replay', false,
        'commerce_noop', true,
        'operation_id', v_operation.id
      );
      update public.tournament_commerce_operations
      set order_id = v_order.id,
          status = 'COMPLETED',
          result_json = v_result,
          completed_at = now(),
          updated_at = now()
      where id = v_operation.id;
      insert into public.tournament_commerce_audit_log (
        club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
        action, before_json, after_json, source
      )
      values (
        p_club_id, p_tournament_id, v_order.id, v_operation.id,
        p_actor_type, p_actor_label, 'ORDER_APPLY_UNCHANGED',
        v_before, v_result, p_source
      );
      return v_result;
    end if;
    if exists (
      select 1
      from public.tournament_commerce_fulfillment fulfillment
      join public.tournament_commerce_order_lines line
        on line.id = fulfillment.order_line_id
      where fulfillment.order_id = v_order.id
        and line.active
        and fulfillment.status = 'FULFILLED'
    ) and v_order.quote_fingerprint <> p_quote_snapshot ->> 'quote_fingerprint' then
      raise exception using errcode = '55000', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLED_ORDER_LOCKED';
    end if;
    update public.tournament_commerce_order_lines
      set active = false where order_id = v_order.id and active;
    update public.tournament_commerce_inventory_reservations
      set status = 'RELEASED', released_at = now(), release_reason = 'ORDER_REVISED'
      where order_id = v_order.id and status = 'ACTIVE';
    update public.tournament_commerce_promotion_claims
      set status = 'RELEASED', released_at = now(), release_reason = 'ORDER_REVISED'
      where order_id = v_order.id and status = 'ACTIVE';
    update public.tournament_commerce_fulfillment
      set status = 'CANCELLED', updated_at = now()
      where order_id = v_order.id and status <> 'FULFILLED';
    v_revision := v_order.current_revision + 1;
  else
    v_before := null;
    v_revision := 1;
    insert into public.tournament_commerce_orders (
      tournament_id, club_id, registration_id, status, payment_status, currency,
      current_revision, request_fingerprint, quote_fingerprint,
      list_subtotal_minor, discount_minor, total_minor
    )
    values (
      p_tournament_id, p_club_id, p_registration_id, 'ACTIVE',
      case upper(coalesce(v_registration.payment_status, 'UNPAID'))
        when 'PAID' then 'PAID'
        when 'WAIVED' then 'WAIVED'
        when 'REFUNDED' then 'REFUNDED'
        else 'UNPAID'
      end,
      'USD',
      1, p_request_fingerprint, p_quote_snapshot ->> 'quote_fingerprint',
      v_list_total, v_discount, v_total
    )
    returning * into v_order;
  end if;

  if v_revision > 1 then
    update public.tournament_commerce_orders
    set status = 'ACTIVE',
        current_revision = v_revision,
        request_fingerprint = p_request_fingerprint,
        quote_fingerprint = p_quote_snapshot ->> 'quote_fingerprint',
        list_subtotal_minor = v_list_total,
        discount_minor = v_discount,
        total_minor = v_total,
        cancelled_at = null,
        cancellation_reason = null,
        updated_at = now()
    where id = v_order.id
    returning * into v_order;
  end if;

  update public.tournament_commerce_operations
  set order_id = v_order.id, updated_at = now()
  where id = v_operation.id;

  insert into public.tournament_commerce_order_revisions (
    order_id, tournament_id, registration_id, revision,
    request_fingerprint, quote_fingerprint, catalog_fingerprint, currency,
    list_subtotal_minor, discount_minor, total_minor, price_snapshot,
    actor_type, actor_label, source
  )
  values (
    v_order.id, p_tournament_id, p_registration_id, v_revision,
    p_request_fingerprint, p_quote_snapshot ->> 'quote_fingerprint',
    p_quote_snapshot ->> 'catalog_fingerprint', 'USD',
    v_list_total, v_discount, v_total, p_quote_snapshot,
    p_actor_type, p_actor_label, p_source
  )
  returning id into v_revision_id;

  -- Lock both global item pools and option-level pools in stable order before
  -- checking availability.  An item limit is shared across every size/variant.
  perform item.id
  from public.tournament_commerce_items item
  where item.id in (
    select (inventory ->> 'item_id')::uuid
    from jsonb_array_elements(p_quote_snapshot -> 'inventory') inventory
  )
  order by item.id
  for update;
  perform variant.id
  from public.tournament_commerce_item_variants variant
  where variant.id in (
    select (inventory ->> 'variant_id')::uuid
    from jsonb_array_elements(p_quote_snapshot -> 'inventory') inventory
  )
  order by variant.id
  for update;

  for v_inventory in
    select value from jsonb_array_elements(p_quote_snapshot -> 'inventory')
  loop
    select variant as variant_row, item as item_row
    into v_pair
    from public.tournament_commerce_item_variants variant
    join public.tournament_commerce_items item on item.id = variant.item_id
    where variant.id = (v_inventory ->> 'variant_id')::uuid
      and item.id = (v_inventory ->> 'item_id')::uuid
      and item.tournament_id = p_tournament_id
      and item.club_id = p_club_id
      and item.status = 'ACTIVE'
      and variant.status = 'ACTIVE';
    if not found then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_INVENTORY_TARGET_INVALID';
    end if;
    v_variant := v_pair.variant_row;
    v_item := v_pair.item_row;
    select coalesce(sum(reservation.quantity), 0)
    into v_reserved
    from public.tournament_commerce_inventory_reservations reservation
    where reservation.variant_id = v_variant.id
      and reservation.status = 'ACTIVE'
      and reservation.order_id <> v_order.id;
    select coalesce(sum(reservation.quantity), 0)
    into v_item_reserved
    from public.tournament_commerce_inventory_reservations reservation
    where reservation.item_id = v_item.id
      and reservation.status = 'ACTIVE'
      and reservation.order_id <> v_order.id;
    if v_variant.inventory_limit is not null
       and v_reserved + (v_inventory ->> 'quantity')::integer
           > v_variant.inventory_limit then
      raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_INVENTORY_UNAVAILABLE';
    end if;
    if v_item.inventory_limit is not null
       and v_item_reserved + (
         select coalesce(sum((requested ->> 'quantity')::integer), 0)
         from jsonb_array_elements(p_quote_snapshot -> 'inventory') requested
         where requested ->> 'item_id' = v_item.id::text
       ) > v_item.inventory_limit then
      raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_INVENTORY_UNAVAILABLE';
    end if;
    insert into public.tournament_commerce_inventory_reservations (
      order_id, revision_id, tournament_id, item_id, variant_id, quantity
    )
    values (
      v_order.id, v_revision_id, p_tournament_id, v_item.id, v_variant.id,
      (v_inventory ->> 'quantity')::integer
    );
  end loop;

  -- Lock and validate giveaway quotas in stable id order.
  perform promotion.id
  from public.tournament_commerce_promotions promotion
  where promotion.id in (
    select (promo ->> 'promotion_id')::uuid
    from jsonb_array_elements(p_quote_snapshot -> 'applied_promotions') promo
  )
  order by promotion.id
  for update;

  for v_promo in
    select value from jsonb_array_elements(p_quote_snapshot -> 'applied_promotions')
  loop
    select promotion.* into v_promotion
    from public.tournament_commerce_promotions promotion
    where promotion.id = (v_promo ->> 'promotion_id')::uuid
      and promotion.tournament_id = p_tournament_id
      and promotion.club_id = p_club_id
      and promotion.status = 'ACTIVE';
    if not found then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_PROMOTION_INVALID';
    end if;
    if (v_promo ->> 'quantity')::integer > v_promotion.per_registration_limit then
      raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_PROMOTION_LIMIT';
    end if;
    if v_promotion.promotion_type = 'DATE_WINDOW_FREE'
       and (
         v_registration.submitted_at < v_promotion.starts_at
         or v_registration.submitted_at > v_promotion.ends_at
       ) then
      raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_PROMOTION_WINDOW_CLOSED';
    elsif v_promotion.promotion_type = 'FIRST_N_REGISTRANTS' then
      select ranked.position into v_rank
      from (
        select registration.id,
               row_number() over (order by registration.submitted_at, registration.id) as position
        from public.tournament_registrations registration
        where registration.tournament_id = p_tournament_id::text
          and lower(coalesce(registration.status, 'confirmed')) <> 'cancelled'
      ) ranked
      where ranked.id = p_registration_id;
      if v_rank is null or v_rank > v_promotion.giveaway_limit then
        raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRANT_GIVEAWAY_FULL';
      end if;
    elsif v_promotion.promotion_type = 'FIRST_N_CLAIMS' then
      select coalesce(sum(claim.quantity), 0) into v_claimed
      from public.tournament_commerce_promotion_claims claim
      where claim.promotion_id = v_promotion.id
        and claim.status = 'ACTIVE'
        and claim.order_id <> v_order.id;
      if v_claimed + (v_promo ->> 'quantity')::integer > v_promotion.giveaway_limit then
        raise exception using errcode = '23514', message = 'JUPR_TOURNAMENT_COMMERCE_CLAIM_GIVEAWAY_FULL';
      end if;
    end if;
    insert into public.tournament_commerce_promotion_claims (
      order_id, revision_id, registration_id, promotion_id, quantity
    )
    values (
      v_order.id, v_revision_id, p_registration_id, v_promotion.id,
      (v_promo ->> 'quantity')::integer
    );
  end loop;

  for v_line in
    select value from jsonb_array_elements(p_quote_snapshot -> 'lines')
  loop
    insert into public.tournament_commerce_order_lines (
      order_id, revision_id, tournament_id, line_key, line_type,
      event_option_id, item_id, variant_id, bundle_id, promotion_id,
      label_snapshot, option_snapshot, quantity, list_unit_minor,
      final_unit_minor, list_total_minor, final_total_minor,
      requires_fulfillment, line_snapshot
    )
    values (
      v_order.id,
      v_revision_id,
      p_tournament_id,
      v_line ->> 'line_key',
      v_line ->> 'line_type',
      nullif(v_line ->> 'event_option_id', ''),
      nullif(v_line ->> 'item_id', '')::uuid,
      nullif(v_line ->> 'variant_id', '')::uuid,
      nullif(v_line ->> 'bundle_id', '')::uuid,
      nullif(v_line ->> 'promotion_id', '')::uuid,
      v_line ->> 'label',
      nullif(v_line ->> 'option_label', ''),
      (v_line ->> 'quantity')::integer,
      (v_line ->> 'list_unit_minor')::integer,
      (v_line ->> 'final_unit_minor')::integer,
      (v_line ->> 'list_total_minor')::integer,
      (v_line ->> 'final_total_minor')::integer,
      coalesce((v_line ->> 'requires_fulfillment')::boolean, false),
      v_line
    )
    returning id into v_line_id;
    if v_line ->> 'line_type' = 'ITEM'
       and coalesce((v_line ->> 'requires_fulfillment')::boolean, false) then
      insert into public.tournament_commerce_fulfillment (
        order_id, revision_id, order_line_id, tournament_id,
        item_id, variant_id, label_snapshot, option_snapshot, sku_snapshot,
        instructions_snapshot, quantity
      )
      select
        v_order.id, v_revision_id, v_line_id, p_tournament_id,
        item.id, variant.id, item.name, variant.name, variant.sku,
        item.fulfillment_instructions, (v_line ->> 'quantity')::integer
      from public.tournament_commerce_item_variants variant
      join public.tournament_commerce_items item on item.id = variant.item_id
      where variant.id = (v_line ->> 'variant_id')::uuid
        and item.id = (v_line ->> 'item_id')::uuid
        and item.tournament_id = p_tournament_id
        and item.club_id = p_club_id
        and item.requires_fulfillment;
      if not found then
        raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLMENT_TARGET_INVALID';
      end if;
    elsif v_line ->> 'line_type' = 'BUNDLE'
          and coalesce((v_line ->> 'requires_fulfillment')::boolean, false) then
      for v_component in
        select jsonb_build_object(
          'item_id', item.id,
          'variant_id', variant.id,
          'label', item.name,
          'option_label', variant.name,
          'sku', variant.sku,
          'instructions', item.fulfillment_instructions,
          'quantity', component.quantity * (v_line ->> 'quantity')::integer
        )
        from public.tournament_commerce_bundle_components component
        join public.tournament_commerce_item_variants variant
          on variant.id = component.variant_id
        join public.tournament_commerce_items item
          on item.id = component.item_id
         and item.id = variant.item_id
        where component.bundle_id = (v_line ->> 'bundle_id')::uuid
          and component.component_type = 'ITEM_VARIANT'
          and item.tournament_id = p_tournament_id
          and item.club_id = p_club_id
          and item.requires_fulfillment
        order by component.id
      loop
        insert into public.tournament_commerce_fulfillment (
          order_id, revision_id, order_line_id, tournament_id,
          item_id, variant_id, label_snapshot, option_snapshot, sku_snapshot,
          instructions_snapshot, quantity
        )
        values (
          v_order.id, v_revision_id, v_line_id, p_tournament_id,
          (v_component ->> 'item_id')::uuid,
          (v_component ->> 'variant_id')::uuid,
          v_component ->> 'label',
          nullif(v_component ->> 'option_label', ''),
          coalesce(v_component ->> 'sku', ''),
          coalesce(v_component ->> 'instructions', ''),
          (v_component ->> 'quantity')::integer
        );
      end loop;
    end if;
  end loop;

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'tournament_commerce_order_apply',
    'order_id', v_order.id,
    'revision_id', v_revision_id,
    'revision', v_revision,
    'registration_id', p_registration_id,
    'quote_fingerprint', p_quote_snapshot ->> 'quote_fingerprint',
    'total_minor', v_total,
    'payment_status', v_order.payment_status,
    'idempotent_replay', false,
    'operation_id', v_operation.id
  );
  update public.tournament_commerce_operations
  set status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    p_club_id, p_tournament_id, v_order.id, v_operation.id,
    p_actor_type, p_actor_label, 'ORDER_APPLY', v_before, v_result, p_source
  );
  return v_result;
end
$function$;

create or replace function public.server_cancel_tournament_commerce_order(
  p_club_id text,
  p_tournament_id uuid,
  p_registration_id text,
  p_expected_order_updated_at timestamptz,
  p_reason text,
  p_idempotency_key uuid,
  p_request_fingerprint text,
  p_actor_type text,
  p_actor_label text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_order public.tournament_commerce_orders%rowtype;
  v_current_revision public.tournament_commerce_order_revisions%rowtype;
  v_cancel_revision_id uuid;
  v_cancel_revision integer;
  v_before jsonb;
  v_operation public.tournament_commerce_operations%rowtype;
  v_result jsonb;
begin
  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_idempotency_key, p_request_fingerprint,
    'ORDER_CANCEL', p_actor_type, p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;
  select orders.* into v_order
  from public.tournament_commerce_orders orders
  where orders.registration_id = p_registration_id
    and orders.tournament_id = p_tournament_id
    and orders.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_NOT_FOUND';
  end if;
  if p_expected_order_updated_at is null
     or v_order.updated_at is distinct from p_expected_order_updated_at then
    raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_STALE';
  end if;
  v_before := to_jsonb(v_order);
  if exists (
    select 1 from public.tournament_commerce_fulfillment fulfillment
    join public.tournament_commerce_order_lines line
      on line.id = fulfillment.order_line_id
    where fulfillment.order_id = v_order.id
      and line.active
      and fulfillment.status = 'FULFILLED'
  ) then
    raise exception using errcode = '55000', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLED_ORDER_LOCKED';
  end if;
  select revision.* into v_current_revision
  from public.tournament_commerce_order_revisions revision
  where revision.order_id = v_order.id
    and revision.revision = v_order.current_revision;
  if not found then
    raise exception using errcode = '55000', message = 'JUPR_TOURNAMENT_COMMERCE_REVISION_EVIDENCE_MISSING';
  end if;
  v_cancel_revision := v_order.current_revision + 1;
  update public.tournament_commerce_orders
  set status = 'CANCELLED',
      current_revision = v_cancel_revision,
      request_fingerprint = p_request_fingerprint,
      cancelled_at = now(),
      cancellation_reason = nullif(btrim(p_reason), ''),
      updated_at = now()
  where id = v_order.id
  returning * into v_order;
  update public.tournament_commerce_inventory_reservations
  set status = 'RELEASED', released_at = now(), release_reason = 'ORDER_CANCELLED'
  where order_id = v_order.id and status = 'ACTIVE';
  update public.tournament_commerce_promotion_claims
  set status = 'RELEASED', released_at = now(), release_reason = 'ORDER_CANCELLED'
  where order_id = v_order.id and status = 'ACTIVE';
  update public.tournament_commerce_fulfillment
  set status = 'CANCELLED', updated_at = now()
  where order_id = v_order.id and status <> 'FULFILLED';
  update public.tournament_commerce_order_lines
  set active = false
  where order_id = v_order.id and active;
  insert into public.tournament_commerce_order_revisions (
    order_id, tournament_id, registration_id, revision,
    request_fingerprint, quote_fingerprint, catalog_fingerprint, currency,
    list_subtotal_minor, discount_minor, total_minor, price_snapshot,
    actor_type, actor_label, source
  )
  values (
    v_order.id, p_tournament_id, p_registration_id, v_cancel_revision,
    p_request_fingerprint, v_current_revision.quote_fingerprint,
    v_current_revision.catalog_fingerprint, v_current_revision.currency,
    v_current_revision.list_subtotal_minor, v_current_revision.discount_minor,
    v_current_revision.total_minor,
    v_current_revision.price_snapshot || jsonb_build_object(
      'order_status', 'CANCELLED',
      'cancelled_at', v_order.cancelled_at,
      'cancellation_reason', v_order.cancellation_reason
    ),
    p_actor_type, p_actor_label, p_source
  )
  returning id into v_cancel_revision_id;
  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'tournament_commerce_order_cancel',
    'order_id', v_order.id,
    'registration_id', p_registration_id,
    'status', v_order.status,
    'revision', v_cancel_revision,
    'revision_id', v_cancel_revision_id,
    'idempotent_replay', false,
    'operation_id', v_operation.id
  );
  update public.tournament_commerce_operations
  set order_id = v_order.id,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  ) values (
    p_club_id, p_tournament_id, v_order.id, v_operation.id,
    p_actor_type, p_actor_label, 'ORDER_CANCEL', v_before, v_result, p_source
  );
  return v_result;
end
$function$;

create or replace function private.tournament_commerce_sync_registration_cancel()
returns trigger
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_order public.tournament_commerce_orders%rowtype;
  v_current_revision public.tournament_commerce_order_revisions%rowtype;
  v_cancel_revision integer;
  v_cancel_revision_id uuid;
  v_before jsonb;
  v_request_fingerprint text;
  v_after jsonb;
begin
  select orders.* into v_order
  from public.tournament_commerce_orders orders
  where orders.registration_id = new.id
    and orders.tournament_id::text = new.tournament_id
  for update;
  if not found or v_order.status = 'CANCELLED' then
    return new;
  end if;

  if exists (
    select 1
    from public.tournament_commerce_fulfillment fulfillment
    join public.tournament_commerce_order_lines line
      on line.id = fulfillment.order_line_id
    where fulfillment.order_id = v_order.id
      and line.active
      and fulfillment.status = 'FULFILLED'
  ) then
    raise exception using
      errcode = '55000',
      message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLED_REGISTRATION_CANCEL_LOCKED';
  end if;

  select revision.* into v_current_revision
  from public.tournament_commerce_order_revisions revision
  where revision.order_id = v_order.id
    and revision.revision = v_order.current_revision;
  if not found then
    raise exception using
      errcode = '55000',
      message = 'JUPR_TOURNAMENT_COMMERCE_REVISION_EVIDENCE_MISSING';
  end if;

  v_before := to_jsonb(v_order);
  v_cancel_revision := v_order.current_revision + 1;
  v_request_fingerprint := md5(
    v_order.id::text
    || ':'
    || v_cancel_revision::text
    || ':REGISTRATION_STATUS_CANCELLED'
  );

  update public.tournament_commerce_orders
  set status = 'CANCELLED',
      current_revision = v_cancel_revision,
      request_fingerprint = v_request_fingerprint,
      cancelled_at = now(),
      cancellation_reason = 'Registration status changed to cancelled',
      updated_at = now()
  where id = v_order.id
  returning * into v_order;

  update public.tournament_commerce_inventory_reservations
  set status = 'RELEASED',
      released_at = now(),
      release_reason = 'REGISTRATION_CANCELLED'
  where order_id = v_order.id and status = 'ACTIVE';
  update public.tournament_commerce_promotion_claims
  set status = 'RELEASED',
      released_at = now(),
      release_reason = 'REGISTRATION_CANCELLED'
  where order_id = v_order.id and status = 'ACTIVE';
  update public.tournament_commerce_fulfillment
  set status = 'CANCELLED', updated_at = now()
  where order_id = v_order.id and status <> 'FULFILLED';
  update public.tournament_commerce_order_lines
  set active = false
  where order_id = v_order.id and active;

  insert into public.tournament_commerce_order_revisions (
    order_id, tournament_id, registration_id, revision,
    request_fingerprint, quote_fingerprint, catalog_fingerprint, currency,
    list_subtotal_minor, discount_minor, total_minor, price_snapshot,
    actor_type, actor_label, source
  )
  values (
    v_order.id, v_order.tournament_id, new.id, v_cancel_revision,
    v_request_fingerprint, v_current_revision.quote_fingerprint,
    v_current_revision.catalog_fingerprint, v_current_revision.currency,
    v_current_revision.list_subtotal_minor, v_current_revision.discount_minor,
    v_current_revision.total_minor,
    v_current_revision.price_snapshot || jsonb_build_object(
      'order_status', 'CANCELLED',
      'cancelled_at', v_order.cancelled_at,
      'cancellation_reason', v_order.cancellation_reason,
      'registration_status', new.status
    ),
    'SYSTEM_RECOVERY',
    'registration-status-sync',
    'registration_status_cancel_sync'
  )
  returning id into v_cancel_revision_id;

  v_after := jsonb_build_object(
    'order_id', v_order.id,
    'registration_id', new.id,
    'order_status', v_order.status,
    'registration_status', new.status,
    'revision', v_cancel_revision,
    'revision_id', v_cancel_revision_id,
    'inventory_and_claims_released', true
  );
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    v_order.club_id, v_order.tournament_id, v_order.id,
    'SYSTEM_RECOVERY', 'registration-status-sync',
    'REGISTRATION_STATUS_CANCEL_SYNC', v_before, v_after,
    'registration_status_cancel_sync'
  );
  return new;
end
$function$;

revoke all on function private.tournament_commerce_sync_registration_cancel()
  from public, anon, authenticated;

drop trigger if exists tournament_commerce_registration_cancel_sync
  on public.tournament_registrations;
create trigger tournament_commerce_registration_cancel_sync
before update of status on public.tournament_registrations
for each row
when (
  lower(coalesce(new.status, '')) = 'cancelled'
  and lower(coalesce(old.status, '')) <> 'cancelled'
)
execute function private.tournament_commerce_sync_registration_cancel();

create or replace function public.server_update_tournament_commerce_fulfillment(
  p_club_id text,
  p_tournament_id uuid,
  p_fulfillment_id uuid,
  p_status text,
  p_notes text,
  p_expected_updated_at timestamptz,
  p_idempotency_key uuid,
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
  v_fulfillment public.tournament_commerce_fulfillment%rowtype;
  v_order public.tournament_commerce_orders%rowtype;
  v_before jsonb;
  v_operation public.tournament_commerce_operations%rowtype;
  v_result jsonb;
  v_pair record;
begin
  if p_status not in ('PENDING', 'READY', 'FULFILLED', 'CANCELLED') then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLMENT_STATUS_INVALID';
  end if;
  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_idempotency_key, p_request_fingerprint,
    'FULFILLMENT_UPDATE', 'ADMIN', p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;
  select fulfillment as fulfillment_row, orders as order_row
  into v_pair
  from public.tournament_commerce_fulfillment fulfillment
  join public.tournament_commerce_orders orders on orders.id = fulfillment.order_id
  where fulfillment.id = p_fulfillment_id
    and fulfillment.tournament_id = p_tournament_id
    and orders.club_id = p_club_id
  for update of fulfillment, orders;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLMENT_NOT_FOUND';
  end if;
  v_fulfillment := v_pair.fulfillment_row;
  v_order := v_pair.order_row;
  if p_expected_updated_at is null
     or v_fulfillment.updated_at is distinct from p_expected_updated_at then
    raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLMENT_STALE';
  end if;
  if v_order.status = 'CANCELLED' and p_status <> 'CANCELLED' then
    raise exception using errcode = '55000', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_CANCELLED';
  end if;
  if v_fulfillment.status = 'FULFILLED'
     and p_status <> 'FULFILLED'
     and length(btrim(coalesce(p_notes, ''))) < 8 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMMERCE_FULFILLED_CORRECTION_NOTE_REQUIRED';
  end if;
  v_before := to_jsonb(v_fulfillment);
  update public.tournament_commerce_fulfillment
  set status = p_status,
      notes = coalesce(p_notes, ''),
      fulfilled_at = case when p_status = 'FULFILLED' then now() else null end,
      fulfilled_by = case when p_status = 'FULFILLED' then p_actor_label else null end,
      updated_at = now()
  where id = p_fulfillment_id
  returning * into v_fulfillment;
  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'tournament_commerce_fulfillment_update',
    'fulfillment_id', v_fulfillment.id,
    'order_id', v_order.id,
    'status', v_fulfillment.status,
    'updated_at', v_fulfillment.updated_at,
    'idempotent_replay', false,
    'operation_id', v_operation.id
  );
  update public.tournament_commerce_operations
  set order_id = v_order.id,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  ) values (
    p_club_id, p_tournament_id, v_order.id, v_operation.id,
    'ADMIN', p_actor_label, 'FULFILLMENT_UPDATE', v_before, to_jsonb(v_fulfillment), p_source
  );
  return v_result;
end
$function$;

create or replace function public.server_update_tournament_commerce_payment(
  p_club_id text,
  p_tournament_id uuid,
  p_registration_id text,
  p_payment_status text,
  p_expected_order_updated_at timestamptz,
  p_idempotency_key uuid,
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
  v_order public.tournament_commerce_orders%rowtype;
  v_before jsonb;
  v_operation public.tournament_commerce_operations%rowtype;
  v_result jsonb;
begin
  if p_payment_status not in ('UNPAID', 'PAID', 'WAIVED', 'REFUNDED') then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_PAYMENT_STATUS_INVALID';
  end if;
  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_idempotency_key, p_request_fingerprint,
    'PAYMENT_UPDATE', 'ADMIN', p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;
  select orders.* into v_order
  from public.tournament_commerce_orders orders
  where orders.registration_id = p_registration_id
    and orders.tournament_id = p_tournament_id
    and orders.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_NOT_FOUND';
  end if;
  if p_expected_order_updated_at is null
     or v_order.updated_at is distinct from p_expected_order_updated_at then
    raise exception using errcode = '40001', message = 'JUPR_TOURNAMENT_COMMERCE_ORDER_STALE';
  end if;
  v_before := to_jsonb(v_order);
  update public.tournament_commerce_orders
  set payment_status = p_payment_status, updated_at = now()
  where id = v_order.id
  returning * into v_order;
  update public.tournament_registrations
  set payment_status = lower(p_payment_status),
      updated_at = now()
  where id = p_registration_id
    and tournament_id = p_tournament_id::text;
  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'tournament_commerce_payment_update',
    'order_id', v_order.id,
    'registration_id', p_registration_id,
    'payment_status', v_order.payment_status,
    'updated_at', v_order.updated_at,
    'idempotent_replay', false,
    'operation_id', v_operation.id
  );
  update public.tournament_commerce_operations
  set order_id = v_order.id,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  ) values (
    p_club_id, p_tournament_id, v_order.id, v_operation.id,
    'ADMIN', p_actor_label, 'PAYMENT_UPDATE', v_before, to_jsonb(v_order), p_source
  );
  return v_result;
end
$function$;

-- Public registration orchestration stays inside one database transaction.
-- The outer receipt is bound to the complete registration/edit request and is
-- checked before duplicate-email or edit-version guards.  This makes a
-- response-loss retry return the committed result instead of leaving a
-- registration without its order.
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
  v_operation public.tournament_commerce_operations%rowtype;
  v_tournament public.tournaments%rowtype;
  v_registration_id text;
  v_submitted_at timestamptz;
  v_selection_count integer;
  v_order_result jsonb;
  v_result jsonb;
begin
  select tournament.* into v_tournament
  from public.tournaments tournament
  where tournament.id = p_tournament_id
    and tournament.club_id = p_club_id
  for update;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_TOURNAMENT_NOT_FOUND';
  end if;

  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_operation_idempotency_key,
    p_request_fingerprint, 'REGISTRATION_CREATE_WITH_COMMERCE',
    'PUBLIC_REGISTRANT', p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;

  v_registration_id := nullif(btrim(p_registration ->> 'id'), '');
  if v_registration_id is null
     or p_registration is null
     or jsonb_typeof(p_registration) <> 'object'
     or p_selections is null
     or jsonb_typeof(p_selections) <> 'array'
     or jsonb_array_length(p_selections) = 0
     or p_quote_snapshot is null
     or jsonb_typeof(p_quote_snapshot) <> 'object'
     or nullif(btrim(p_registration ->> 'tournament_id'), '') <> p_tournament_id::text
     or nullif(lower(btrim(p_registration ->> 'email')), '') is null
     or nullif(btrim(p_registration ->> 'display_name'), '') is null then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_INVALID';
  end if;

  if exists (
    select 1
    from public.tournament_registrations registration
    where registration.tournament_id = p_tournament_id::text
      and lower(btrim(registration.email))
          = lower(btrim(p_registration ->> 'email'))
  ) then
    raise exception using errcode = '23505', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_DUPLICATE';
  end if;

  if (
    select coalesce(array_agg(distinct event_id order by event_id), array[]::text[])
    from jsonb_array_elements_text(
      coalesce(p_quote_snapshot -> 'request' -> 'event_option_ids', '[]'::jsonb)
    ) as quote_event(event_id)
  ) is distinct from (
    select coalesce(
      array_agg(distinct btrim(selection.event_option_id) order by btrim(selection.event_option_id)),
      array[]::text[]
    )
    from jsonb_to_recordset(p_selections) selection(event_option_id text)
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_EVENT_MISMATCH';
  end if;

  if exists (
    select 1
    from jsonb_to_recordset(p_selections) selection(
      id text,
      tournament_id text,
      registration_id text,
      registration_day_id text,
      event_option_id text,
      partner_mode text
    )
    left join public.tournament_registration_days day
      on day.id = selection.registration_day_id
     and day.tournament_id = p_tournament_id::text
    left join public.tournament_event_options event
      on event.id = selection.event_option_id
     and event.tournament_id = p_tournament_id::text
     and event.registration_day_id = selection.registration_day_id
    where nullif(btrim(selection.id), '') is null
       or selection.tournament_id <> p_tournament_id::text
       or selection.registration_id <> v_registration_id
       or day.id is null
       or event.id is null
       or not coalesce(day.enabled, false)
       or not coalesce(event.enabled, false)
       or lower(coalesce(nullif(btrim(event.status), ''), 'draft'))
          not in ('open', 'tentative', 'confirmed')
       or upper(coalesce(nullif(btrim(selection.partner_mode), ''), 'NONE'))
          not in ('NONE', 'HAS_PARTNER', 'NEEDS_PARTNER')
  ) or (
    select count(*)
    from jsonb_to_recordset(p_selections) selection(id text)
  ) <> (
    select count(distinct btrim(selection.id))
    from jsonb_to_recordset(p_selections) selection(id text)
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_SELECTION_INVALID';
  end if;

  v_submitted_at := pg_catalog.clock_timestamp();
  insert into public.tournament_registrations (
    id, tournament_id, submitted_at, updated_at, status, payment_status,
    first_name, last_name, display_name, email, phone, player_id, dupr_id,
    doubles_skill, singles_skill, age, age_bracket, gender, notes,
    wants_partner_board_contact
  )
  values (
    v_registration_id,
    p_tournament_id::text,
    v_submitted_at,
    v_submitted_at,
    lower(coalesce(nullif(btrim(p_registration ->> 'status'), ''), 'confirmed')),
    lower(coalesce(nullif(btrim(p_registration ->> 'payment_status'), ''), 'unpaid')),
    nullif(btrim(p_registration ->> 'first_name'), ''),
    nullif(btrim(p_registration ->> 'last_name'), ''),
    btrim(p_registration ->> 'display_name'),
    lower(btrim(p_registration ->> 'email')),
    nullif(btrim(p_registration ->> 'phone'), ''),
    nullif(p_registration ->> 'player_id', '')::integer,
    nullif(btrim(p_registration ->> 'dupr_id'), ''),
    nullif(p_registration ->> 'doubles_skill', '')::numeric,
    nullif(p_registration ->> 'singles_skill', '')::numeric,
    nullif(p_registration ->> 'age', '')::integer,
    nullif(btrim(p_registration ->> 'age_bracket'), ''),
    nullif(btrim(p_registration ->> 'gender'), ''),
    nullif(btrim(p_registration ->> 'notes'), ''),
    coalesce((p_registration ->> 'wants_partner_board_contact')::boolean, false)
  );

  insert into public.tournament_registration_selections (
    id, tournament_id, registration_id, registration_day_id,
    event_option_id, partner_mode, partner_name, partner_email,
    partner_phone, partner_dupr_id, partner_skill, partner_age,
    partner_note, show_on_partner_board, sort_order, created_at, updated_at
  )
  select
    selection.id,
    p_tournament_id::text,
    v_registration_id,
    selection.registration_day_id,
    selection.event_option_id,
    upper(coalesce(nullif(btrim(selection.partner_mode), ''), 'NONE')),
    nullif(btrim(selection.partner_name), ''),
    nullif(lower(btrim(selection.partner_email)), ''),
    nullif(btrim(selection.partner_phone), ''),
    nullif(btrim(selection.partner_dupr_id), ''),
    selection.partner_skill,
    selection.partner_age,
    nullif(btrim(selection.partner_note), ''),
    coalesce(selection.show_on_partner_board, false),
    selection.sort_order,
    v_submitted_at,
    v_submitted_at
  from jsonb_to_recordset(p_selections) selection(
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
    partner_note text,
    show_on_partner_board boolean,
    sort_order integer
  );
  get diagnostics v_selection_count = row_count;

  v_order_result := public.server_apply_tournament_commerce_order(
    p_club_id,
    p_tournament_id,
    v_registration_id,
    null,
    p_quote_snapshot,
    p_order_idempotency_key,
    p_quote_snapshot ->> 'request_fingerprint',
    'PUBLIC_REGISTRANT',
    p_actor_label,
    p_source
  );

  v_result := jsonb_build_object(
    'ok', true,
    'mode', 'registration_create_with_commerce',
    'registration_id', v_registration_id,
    'submitted_at', v_submitted_at,
    'updated_at', v_submitted_at,
    'selection_count', v_selection_count,
    'commerce_order', v_order_result,
    'operation_id', v_operation.id,
    'idempotent_replay', false
  );
  update public.tournament_commerce_operations
  set order_id = nullif(v_order_result ->> 'order_id', '')::uuid,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    p_club_id, p_tournament_id,
    nullif(v_order_result ->> 'order_id', '')::uuid, v_operation.id,
    'PUBLIC_REGISTRANT', p_actor_label,
    'REGISTRATION_CREATE_WITH_COMMERCE', null, v_result, p_source
  );
  return v_result;
end
$function$;

create or replace function public.server_update_public_tournament_registration_with_commerce(
  p_club_id text,
  p_tournament_id uuid,
  p_registration_id text,
  p_expected_registration_updated_at timestamptz,
  p_expected_selection_versions jsonb,
  p_registration_patch jsonb,
  p_selections jsonb,
  p_expected_order_updated_at timestamptz,
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
  v_operation public.tournament_commerce_operations%rowtype;
  v_edit_result jsonb;
  v_order_result jsonb;
  v_result jsonb;
begin
  v_operation := private.tournament_commerce_operation_begin(
    p_club_id, p_tournament_id, p_operation_idempotency_key,
    p_request_fingerprint, 'REGISTRATION_EDIT_WITH_COMMERCE',
    'PUBLIC_REGISTRANT', p_actor_label, p_source
  );
  if v_operation.status = 'COMPLETED' then
    return v_operation.result_json || jsonb_build_object('idempotent_replay', true);
  end if;

  if (
    select coalesce(array_agg(distinct event_id order by event_id), array[]::text[])
    from jsonb_array_elements_text(
      coalesce(p_quote_snapshot -> 'request' -> 'event_option_ids', '[]'::jsonb)
    ) as quote_event(event_id)
  ) is distinct from (
    select coalesce(
      array_agg(distinct btrim(selection.event_option_id) order by btrim(selection.event_option_id)),
      array[]::text[]
    )
    from jsonb_to_recordset(p_selections) selection(event_option_id text)
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_EVENT_MISMATCH';
  end if;

  v_edit_result := public.server_update_public_tournament_registration_edit(
    p_tournament_id::text,
    p_registration_id,
    p_expected_registration_updated_at,
    p_expected_selection_versions,
    p_registration_patch,
    p_selections
  );
  if coalesce((v_edit_result ->> 'ok')::boolean, false) is not true then
    if v_edit_result ->> 'code' = 'REGISTRATION_EDIT_CONFLICT' then
      raise exception using errcode = '40001', message = 'JUPR_REGISTRATION_EDIT_CONFLICT';
    elsif v_edit_result ->> 'code' = 'REGISTRATION_IMPORTED_TO_DRAW' then
      raise exception using errcode = '55000', message = 'JUPR_REGISTRATION_IMPORTED_TO_DRAW';
    elsif v_edit_result ->> 'code' = 'REGISTRATION_RELATIONSHIP_LOCKED' then
      raise exception using errcode = '55000', message = 'JUPR_REGISTRATION_RELATIONSHIP_LOCKED';
    end if;
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_COMMERCE_REGISTRATION_EDIT_INVALID';
  end if;

  v_order_result := public.server_apply_tournament_commerce_order(
    p_club_id,
    p_tournament_id,
    p_registration_id,
    p_expected_order_updated_at,
    p_quote_snapshot,
    p_order_idempotency_key,
    p_quote_snapshot ->> 'request_fingerprint',
    'PUBLIC_REGISTRANT',
    p_actor_label,
    p_source
  );
  v_result := v_edit_result || jsonb_build_object(
    'mode', 'registration_edit_with_commerce',
    'commerce_order', v_order_result,
    'operation_id', v_operation.id,
    'idempotent_replay', false
  );
  update public.tournament_commerce_operations
  set order_id = nullif(v_order_result ->> 'order_id', '')::uuid,
      status = 'COMPLETED',
      result_json = v_result,
      completed_at = now(),
      updated_at = now()
  where id = v_operation.id;
  insert into public.tournament_commerce_audit_log (
    club_id, tournament_id, order_id, operation_id, actor_type, actor_label,
    action, before_json, after_json, source
  )
  values (
    p_club_id, p_tournament_id,
    nullif(v_order_result ->> 'order_id', '')::uuid, v_operation.id,
    'PUBLIC_REGISTRANT', p_actor_label,
    'REGISTRATION_EDIT_WITH_COMMERCE',
    jsonb_build_object(
      'expected_registration_updated_at', p_expected_registration_updated_at,
      'expected_order_updated_at', p_expected_order_updated_at
    ),
    v_result, p_source
  );
  return v_result;
end
$function$;

revoke all on function public.server_replace_tournament_commerce_catalog(
  text, uuid, text, jsonb, uuid, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_apply_tournament_commerce_order(
  text, uuid, text, timestamptz, jsonb, uuid, text, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_cancel_tournament_commerce_order(
  text, uuid, text, timestamptz, text, uuid, text, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_update_tournament_commerce_fulfillment(
  text, uuid, uuid, text, text, timestamptz, uuid, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_update_tournament_commerce_payment(
  text, uuid, text, text, timestamptz, uuid, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_create_public_tournament_registration_with_commerce(
  text, uuid, jsonb, jsonb, jsonb, uuid, uuid, text, text, text
) from public, anon, authenticated;
revoke all on function public.server_update_public_tournament_registration_with_commerce(
  text, uuid, text, timestamptz, jsonb, jsonb, jsonb, timestamptz,
  jsonb, uuid, uuid, text, text, text
) from public, anon, authenticated;

grant execute on function public.server_replace_tournament_commerce_catalog(
  text, uuid, text, jsonb, uuid, text, text, text
) to service_role;
grant execute on function public.server_apply_tournament_commerce_order(
  text, uuid, text, timestamptz, jsonb, uuid, text, text, text, text
) to service_role;
grant execute on function public.server_cancel_tournament_commerce_order(
  text, uuid, text, timestamptz, text, uuid, text, text, text, text
) to service_role;
grant execute on function public.server_update_tournament_commerce_fulfillment(
  text, uuid, uuid, text, text, timestamptz, uuid, text, text, text
) to service_role;
grant execute on function public.server_update_tournament_commerce_payment(
  text, uuid, text, text, timestamptz, uuid, text, text, text
) to service_role;
grant execute on function public.server_create_public_tournament_registration_with_commerce(
  text, uuid, jsonb, jsonb, jsonb, uuid, uuid, text, text, text
) to service_role;
grant execute on function public.server_update_public_tournament_registration_with_commerce(
  text, uuid, text, timestamptz, jsonb, jsonb, jsonb, timestamptz,
  jsonb, uuid, uuid, text, text, text
) to service_role;

comment on table public.tournament_commerce_order_revisions is
  'Immutable server-priced snapshots retained across registration edits and cancellations.';
comment on table public.tournament_commerce_inventory_reservations is
  'Active/released inventory ledger. Lodging packages use the same fixed-inventory contract.';
comment on table public.tournament_commerce_promotion_claims is
  'Quota ledger for first-N registrant and first-N claim giveaways.';
comment on table public.tournament_commerce_operations is
  'Idempotency and recovery evidence for every commerce mutation.';

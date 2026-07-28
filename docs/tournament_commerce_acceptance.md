# Tournament extras, bundles, and offline payment acceptance

## Scope

Tournament staff can offer optional merchandise, accommodations, meals, drink
packs, and other extras during public registration. Matching event-and-extra
bundles and eligible free offers are applied by the server. Payment remains
offline.

## Staff catalog

- A tournament manager can create, edit, activate, archive, and order extras.
- Every extra has a type, USD base price, description, availability window,
  per-registration limit, optional total inventory, and optional fulfillment
  instructions.
- An extra can have separately priced and inventoried options such as shirt
  size, room type, or meal choice.
- An active extra must have at least one active option.
- A manager can review a complete catalog without writing it.
- Saving requires an exact reviewed catalog fingerprint, a unique idempotency
  key, manager permission, the `SAVE` confirmation, and the isolated
  `tournament-commerce-admin` staging write wave.
- Catalog saves are forward-only revisions. Existing order revisions and audit
  history are never rewritten.

## Bundles

- A manager can define a bundle price, availability window, quantity limit,
  and required combination of tournament events and/or extra options.
- An active bundle must contain at least one valid component.
- The registration quote automatically selects the combination of bundles and
  free offers that produces the greatest valid savings for the complete cart.
- A bundle cannot consume more events or extras than the registrant selected.
- Public review, confirmation, email, and fulfillment show the concrete
  components included in each applied bundle.

## Free offers and giveaways

- A manager can make an item, item option, or bundle free within a start/end
  window.
- A manager can make a limited quantity free for the first N registrants or
  first N valid claims.
- Each offer has a per-registration free limit and deterministic priority.
- Date-window eligibility is bound to the original registration submission
  time when an existing registration is edited.
- First-registrant eligibility is based on authoritative tournament
  registration order.
- First-claim capacity is reserved atomically; concurrent registrations cannot
  receive more free units than the configured limit.
- Cancelling an order releases active inventory and claims without deleting
  their history.

## Public registration

- Registration lists active, currently available extras with prices, options,
  remaining inventory when limited, and maximum quantities.
- The page explains that bundle savings and eligible giveaways apply
  automatically.
- A registrant may select zero extras.
- Before submission, the server returns an authoritative quote covering both
  event fees and extras. The quote displays every line, savings, total due, and
  the fact that payment is handled offline.
- Changing events, extras, inventory-sensitive selections, or the catalog
  invalidates the prior review and rotates the commerce idempotency key.
- A registration submit never accepts a browser-authored price snapshot.
- Submission sends only selections and the expected server quote fingerprint.
  If price, inventory, offer capacity, or catalog state changed, the API returns
  the current quote and requires another explicit review.
- New and edited registrations save the registration, selections, order
  revision, inventory reservations, giveaway claims, fulfillment rows,
  operation evidence, and audit rows in one database transaction.
- A failed transaction leaves none of those records partially written.
- Repeating the same completed idempotency key returns the prior immutable
  result and does not send a duplicate confirmation email.
- Secure edit links can revise extras using the current order version. A stale
  edit receives a conflict instead of overwriting a later order.

## Offline payment, fulfillment, and recovery

- No card details, checkout session, or online charge is created.
- Managers can mark an active order unpaid, paid, waived, or refunded using the
  exact current order version.
- Managers can cancel an order only with a reason and the exact `CANCEL`
  confirmation.
- Managers can track each fulfillable line or bundle component as pending,
  ready, fulfilled, or cancelled.
- Moving a fulfilled row backward requires a correction note of at least eight
  characters.
- Managers can export the current active fulfillment list through an
  authenticated CSV endpoint.
- The staff workspace shows operation state and internal audit history.
  Interrupted or recovery-required work can be inspected against authoritative
  evidence before any retry.
- A completed operation retried with the same key reuses its result and repairs
  a missing secondary admin audit without repeating the domain mutation.

## Permissions and isolation

- Public catalog and quote routes expose only public-safe catalog, inventory,
  and quote data.
- Public commerce writes require the existing `public-intake-auth` wave plus
  `JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES`.
- Staff reads and writes require an authenticated role with
  `PERMISSION_MANAGE_TOURNAMENTS`; denied attempts are recorded for review.
- Staff writes require the separate `tournament-commerce-admin` wave plus the
  same commerce mutation flag.
- All commerce mutation services explicitly reject production.
- Commerce tables use forced row-level security and are server-service-only;
  browser roles receive no direct table or RPC mutation access.
- Staging email redirect and dry-run behavior remain unchanged. Replayed
  registration operations never send a second email.

## Usability and verification

- Public selection and staff editing reflow at normal laptop and mobile widths.
- Wide payment, fulfillment, recovery, and audit tables remain horizontally
  scrollable without hiding actions.
- Async quote and save feedback is announced to assistive technology.
- Focused tests cover pricing, bundle optimization, giveaway limits, inventory,
  atomic registration wrappers, replay/recovery, cancellation, permissions,
  schema isolation, email content, and frontend contract behavior.

-- Only the backend can insert or read cancellation receipts. No public access.
create policy cancellation_receipt_service_read
  on private.tournament_registration_cancellations for select to service_role using (true);
create policy cancellation_receipt_service_insert
  on private.tournament_registration_cancellations for insert to service_role with check (true);

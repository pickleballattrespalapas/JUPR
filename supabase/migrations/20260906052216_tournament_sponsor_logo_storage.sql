-- Logos stay private. Only the authenticated server can upload/sign assets.
-- Public sponsor responses sign only enabled assets from published settings.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('tournament-sponsor-logos', 'tournament-sponsor-logos', false, 5242880, array['image/webp'])
on conflict (id) do nothing;

-- Restrictive policies protect this bucket even if another feature has a broad
-- permissive storage.objects policy. Service-role requests bypass RLS.
create policy "sponsor_logos_no_client_select" on storage.objects as restrictive
for select to anon, authenticated using (bucket_id <> 'tournament-sponsor-logos');
create policy "sponsor_logos_no_client_insert" on storage.objects as restrictive
for insert to anon, authenticated with check (bucket_id <> 'tournament-sponsor-logos');
create policy "sponsor_logos_no_client_update" on storage.objects as restrictive
for update to anon, authenticated using (bucket_id <> 'tournament-sponsor-logos')
with check (bucket_id <> 'tournament-sponsor-logos');
create policy "sponsor_logos_no_client_delete" on storage.objects as restrictive
for delete to anon, authenticated using (bucket_id <> 'tournament-sponsor-logos');

begin;
-- The API cannot read auth.users. Expose only a boolean identity check, with
-- the Auth row locked through the invitation transaction; never return users.
create function public.pcs_staff_verified_email(p_user_id uuid, p_email text)
returns boolean language plpgsql security definer set search_path = '' as $$
declare verified_email text; confirmed_at timestamptz;
begin
 select lower(email),email_confirmed_at into verified_email,confirmed_at
 from auth.users where id=p_user_id for share;
 return coalesce(verified_email=lower(trim(p_email)) and confirmed_at is not null,false);
end $$;
revoke all on function public.pcs_staff_verified_email(uuid,text) from public,anon,authenticated;
grant execute on function public.pcs_staff_verified_email(uuid,text) to service_role;

commit;

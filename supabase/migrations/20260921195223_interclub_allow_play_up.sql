begin;

-- A lower-rated player can enter a higher division. Numeric divisions retain
-- their exact exclusive upper boundary; Open has no rating ceiling. The same
-- helper governs pool summaries, meet rosters and frozen competition lineups.
-- No player ratings, league seeds, pool choices or registration dates change.
create or replace function public.pcs_interclub_rating_in_division(p_rating numeric,p_division text)
returns boolean language plpgsql immutable security invoker set search_path=public as $$
begin
 if p_rating is null or p_rating<=0 or p_rating::text in ('NaN','Infinity','-Infinity') or p_division is null then
  return false;
 end if;
 if lower(p_division) in ('open','4.5/open') then return true; end if;
 if p_division !~ '^[2-6]\.[05]$' then return false; end if;
 return p_rating<p_division::numeric+0.5;
end $$;
revoke all on function public.pcs_interclub_rating_in_division(numeric,text) from public,anon,authenticated;
grant execute on function public.pcs_interclub_rating_in_division(numeric,text) to service_role;

-- Keep installed authorization, phase locks and deadline logic intact. Only
-- the rejected-player guidance changes; both callers use the shared helper.
do $patch$
declare spec record; routine regprocedure; definition text;
begin
 for spec in select * from (values
  ('public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean)',
   'Choose the skill level matching the current interclub rating',
   'Choose a division whose upper rating limit exceeds the current interclub rating'),
  ('public.pcs_assert_interclub_competition_eligibility(uuid,uuid,jsonb,text)',
   'Player must enter the skill level matching the league rating at the roster deadline',
   'Player must be below the division upper rating limit at the roster deadline')
 ) as changes(signature,old_message,new_message) loop
  routine:=to_regprocedure(spec.signature);
  if routine is null then raise exception 'Required interclub eligibility routine missing: %',spec.signature; end if;
  definition:=pg_get_functiondef(routine);
  if (length(definition)-length(replace(definition,spec.old_message,'')))/length(spec.old_message)<>1 then
   raise exception 'Expected one eligibility message in %',spec.signature;
  end if;
  execute replace(definition,spec.old_message,spec.new_message);
 end loop;
end $patch$;

notify pgrst,'reload schema';
commit;

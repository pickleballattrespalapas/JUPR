-- Edition settings and responses are accessible only through the club-scoped API.
create table public.recap_engagement_editions (
    recap_id uuid primary key references public.weekly_recaps(id) on delete cascade,
    club_id text not null,
    week_start date not null,
    scope_key text not null,
    config jsonb not null,
    closes_at timestamptz not null,
    followup_source_id uuid unique,
    followup jsonb,
    unique (club_id, week_start)
);
create index recap_engagement_history on public.recap_engagement_editions (club_id, scope_key, week_start desc);

create table public.recap_engagement_readers (
    id uuid primary key default gen_random_uuid(),
    recap_id uuid not null references public.recap_engagement_editions(recap_id) on delete cascade,
    visitor uuid not null,
    source text not null check (source in ('email','website','whatsapp')),
    email_seen boolean not null default false,
    email_participated boolean not null default false,
    requester_hash text not null,
    created_at timestamptz not null default now(),
    choice smallint check (choice in (0,1)),
    nominee text check (length(nominee) between 1 and 80),
    story text check (length(story) between 1 and 400),
    moderation text check (moderation in ('pending','featured','dismissed')),
    reviewed_by text,
    unique (recap_id, visitor)
);
create index recap_engagement_recent_readers on public.recap_engagement_readers (recap_id, created_at);
alter table public.recap_engagement_editions enable row level security;
alter table public.recap_engagement_readers enable row level security;
revoke all on public.recap_engagement_editions, public.recap_engagement_readers from public, anon, authenticated;
grant select, insert, update, delete on public.recap_engagement_editions, public.recap_engagement_readers to service_role;

create function public.recap_engagement_counts(p_id uuid) returns jsonb
language sql stable security invoker set search_path = '' as $$
    select jsonb_build_object(
        'readers', count(*), 'email_readers', count(*) filter (where email_seen),
        'email_participants', count(*) filter (where email_participated),
        'votes', count(choice), 'first', count(*) filter (where choice=0),
        'second', count(*) filter (where choice=1), 'nominations', count(story))
    from public.recap_engagement_readers where recap_id=p_id
$$;

create function public.sync_weekly_recap_engagement() returns trigger
language plpgsql security invoker set search_path = '' as $$
declare
    settings jsonb := new.published_json #> '{report,engagement}';
    scope_name text := coalesce(nullif(new.published_json #>> '{report,scope_label}',''),'Whole club');
    edition public.recap_engagement_editions%rowtype;
    prior public.recap_engagement_editions%rowtype;
    counts jsonb;
    result jsonb;
    winner int;
    clinic jsonb;
    initial_close timestamptz := now() + interval '7 days';
begin
    if new.status <> 'published' or jsonb_typeof(settings) is distinct from 'object' then return new; end if;
    if tg_op='UPDATE' and old.status='published' and old.published_at is not distinct from new.published_at
       and old.published_json is not distinct from new.published_json then return new; end if;
    settings := settings - 'subject' - 'body';
    perform pg_advisory_xact_lock(hashtextextended('weekly-recap/' || new.club_id, 0));
    select * into edition from public.recap_engagement_editions where recap_id=new.id for update;
    if found then
        if edition.scope_key <> scope_name then raise exception 'A published engagement edition cannot change league scope.'; end if;
        if edition.config->'poll' is distinct from settings->'poll' and exists (
            select 1 from public.recap_engagement_readers where recap_id=new.id and choice is not null
        ) then raise exception 'This poll already has votes. Restore the published question and answers.'; end if;
        update public.recap_engagement_editions set config=settings where recap_id=new.id;
        return new;
    end if;
    if exists (select 1 from public.recap_engagement_editions where club_id=new.club_id
               and scope_key=scope_name and week_start>new.week_start) then
        initial_close := now();
    else
        select e.* into prior from public.recap_engagement_editions e
        join public.weekly_recaps w on w.id=e.recap_id and w.status='published'
        where e.club_id=new.club_id and e.scope_key=scope_name and e.week_start<new.week_start
        order by e.week_start desc limit 1 for update of e;
        update public.recap_engagement_editions set closes_at=least(closes_at,now())
        where club_id=new.club_id and scope_key=scope_name and week_start<new.week_start and closes_at>now();
        if prior.recap_id is not null and not exists (select 1 from public.recap_engagement_editions where followup_source_id=prior.recap_id) then
            counts := public.recap_engagement_counts(prior.recap_id);
            if (counts->>'votes')::int >= 5 then
                winner := case when (counts->>'first')::int > (counts->>'second')::int then 0
                               when (counts->>'second')::int > (counts->>'first')::int then 1 end;
                result := jsonb_build_object('week',prior.week_start,'question',prior.config#>>'{poll,question}',
                    'options',prior.config#>'{poll,options}', 'counts',jsonb_build_array(counts->'first',counts->'second'),
                    'tip',case when winner is not null then prior.config->'poll'->'tips'->>winner end,
                    'winner',winner);
                if winner is not null then
                    select jsonb_build_object('title',item->>'title','url',item->>'url') into clinic
                    from jsonb_array_elements(new.published_json#>'{report,edition}') block
                    cross join lateral jsonb_array_elements(block->'items') item
                    where block->>'key' in ('custom','upcoming') and length(item->>'title')>0
                      and item->>'url' ~ '^https?://[^[:space:]]+$'
                      and item->>'date' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                      and item->>'date' >= greatest(new.week_end,(now() at time zone 'America/Mazatlan')::date)::text
                      and exists (select 1 from regexp_split_to_table(lower(prior.config->'poll'->'options'->>winner),'[^[:alnum:]]+') word
                                  where length(word)>=4 and position(rtrim(word,'s') in lower(concat_ws(' ',item->>'title',item->>'text')))>0)
                    order by item->>'date' limit 1;
                    if clinic is not null then result := result || jsonb_build_object('clinic',clinic); end if;
                end if;
            end if;
        end if;
    end if;
    insert into public.recap_engagement_editions(recap_id,club_id,week_start,scope_key,config,closes_at,followup_source_id,followup)
      values(new.id,new.club_id,new.week_start,scope_name,settings,initial_close,
             case when result is not null then prior.recap_id end,result);
    return new;
end $$;
create trigger weekly_recap_sync_engagement after insert or update of status,published_json,published_at
on public.weekly_recaps for each row execute function public.sync_weekly_recap_engagement();

create function public.record_recap_engagement(p_club text,p_week date,p_visitor uuid,p_source text,
    p_hash text,p_action text,p_question text,p_answer text,p_name text,p_story text) returns jsonb
language plpgsql security invoker set search_path = '' as $$
declare
    e public.recap_engagement_editions%rowtype;
    reader public.recap_engagement_readers%rowtype;
    feature text;
    answer smallint;
begin
    select t.* into e from public.weekly_recaps w join public.recap_engagement_editions t on t.recap_id=w.id
      where w.club_id=p_club and w.week_start=p_week and w.status='published' for share of w for update of t;
    if not found then raise exception 'Recap unavailable' using errcode='P0002'; end if;
    if p_source not in ('website','email','whatsapp') or p_action not in ('visit','vote','shoutout')
       or p_visitor is null or length(p_hash) <> 64 then raise exception 'Invalid response'; end if;
    feature := case when e.config->>'mode'='off' or
       (e.config->>'mode'='review' and not (e.config->>'approved')::boolean) then 'none' else e.config->>'featured' end;
    select * into reader from public.recap_engagement_readers where recap_id=e.recap_id and visitor=p_visitor;
    if not found then
        if (select count(*) from public.recap_engagement_readers where recap_id=e.recap_id and created_at>now()-interval '1 hour'
            and requester_hash=p_hash) >= 240 or (select count(*) from public.recap_engagement_readers where recap_id=e.recap_id
            and created_at>now()-interval '1 hour') >= 2400 then raise exception 'Too many readers'; end if;
        insert into public.recap_engagement_readers(recap_id,visitor,source,requester_hash,email_seen)
            values(e.recap_id,p_visitor,p_source,p_hash,p_source='email') returning * into reader;
    elsif p_source='email' and not reader.email_seen then
        update public.recap_engagement_readers set email_seen=true where id=reader.id returning * into reader;
    end if;
    if p_action='vote' and reader.choice is null then
        if e.closes_at<=now() or feature<>'poll' then raise exception 'Voting is closed'; end if;
        answer := case p_answer when e.config#>>'{poll,options,0}' then 0 when e.config#>>'{poll,options,1}' then 1 end;
        if answer is null or p_question is distinct from e.config#>>'{poll,question}' then raise exception 'Poll changed'; end if;
        update public.recap_engagement_readers set choice=answer,email_participated=email_participated or p_source='email' where id=reader.id returning * into reader;
    elsif p_action='shoutout' and reader.story is null then
        if e.closes_at<=now() or feature='none' or (feature<>'shoutout' and not (e.config->>'shoutouts')::boolean)
        then raise exception 'Nominations are closed'; end if;
        if nullif(trim(p_name),'') is null or nullif(trim(p_story),'') is null then raise exception 'Incomplete nomination'; end if;
        update public.recap_engagement_readers set nominee=trim(p_name),story=trim(p_story),moderation='pending',email_participated=email_participated or p_source='email'
            where id=reader.id returning * into reader;
    end if;
    return jsonb_build_object('choice',reader.choice,'nominated',reader.story is not null);
end $$;
revoke all on function public.sync_weekly_recap_engagement(),public.recap_engagement_counts(uuid),
 public.record_recap_engagement(text,date,uuid,text,text,text,text,text,text,text) from public,anon,authenticated;
grant execute on function public.sync_weekly_recap_engagement(),public.recap_engagement_counts(uuid),
 public.record_recap_engagement(text,date,uuid,text,text,text,text,text,text,text) to service_role;

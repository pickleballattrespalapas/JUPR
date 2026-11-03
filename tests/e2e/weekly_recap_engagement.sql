-- Run against an isolated test database. All fixture changes roll back.
begin;
-- Covers two editions, anonymous receipts, private nominations and repeat publication.
do $$
declare
    club constant text := 'tres_palapas';
    week constant date := '2041-05-06';
    reader_id uuid := gen_random_uuid();
    fingerprint text := repeat('a',64);
    wk1 uuid; wk2 uuid; wk3 uuid; wk4 uuid;
    cfg jsonb := '{"mode":"automatic","featured":"poll","approved":false,"shoutouts":true,"poll":{"id":"test","question":"Which drill?","options":["Drops","Resets"],"tips":["Drop drill","Reset drill"]}}';
    snap jsonb := jsonb_build_object('report',jsonb_build_object('version',1,'scope_label','QA engagement contract','engagement',cfg,'edition','[]'::jsonb));
    counts jsonb; result jsonb;
begin
    insert into public.weekly_recaps(club_id,week_start,week_end,status,final_json) values(club,week,week+6,'draft',snap) returning id into wk1;
    if exists(select from public.recap_engagement_editions where recap_id=wk1) then raise exception 'Draft is already collecting votes'; end if;
    update public.weekly_recaps set status='published',published_at=now(),published_json=snap where id=wk1;
    perform public.record_recap_engagement(club,week,reader_id,'website',fingerprint,'visit','','','','');
    perform public.record_recap_engagement(club,week,reader_id,'email',fingerprint,'vote','Which drill?','Resets','','');
    perform public.record_recap_engagement(club,week,reader_id,'website',fingerprint,'vote','Which drill?','Drops','','');
    perform public.record_recap_engagement(club,week,reader_id,'email',fingerprint,'shoutout','','','QA member','Made room for a new player.');
    if (select moderation from public.recap_engagement_readers where recap_id=wk1 and visitor=reader_id) <> 'pending' then raise exception 'Nomination was not private'; end if;
    begin
        perform public.record_recap_engagement('wrong-club',week,reader_id,'email',fingerprint,'vote','Which drill?','Resets','','');
        raise exception 'cross-club vote accepted';
    exception when no_data_found then null;
    end;
    begin
        update public.weekly_recaps set published_at=clock_timestamp(),published_json=jsonb_set(snap,'{report,engagement,poll,question}','"Different question"') where id=wk1;
        raise exception 'changed a voted-on poll';
    exception when raise_exception then
        if sqlerrm not like 'This poll already has votes.%' then raise; end if;
    end;
    for i in 1..4 loop
        perform public.record_recap_engagement(club,week,gen_random_uuid(),'website',fingerprint,'vote','Which drill?',case when i=1 then 'Drops' else 'Resets' end,'','');
    end loop;
    counts := public.recap_engagement_counts(wk1);
    if counts <> '{"readers":5,"email_readers":1,"email_participants":1,"votes":5,"first":1,"second":4,"nominations":1}'::jsonb then raise exception 'Wrong counts %',counts; end if;
    snap := jsonb_set(snap,'{report,edition}',jsonb_build_array(jsonb_build_object('key','custom','items',jsonb_build_array(
        jsonb_build_object('date','2041-12-31','title','Reset clinic','url','https://example.invalid/qa-reset')))));
    insert into public.weekly_recaps(club_id,week_start,week_end,status,final_json,published_json,published_at)
      values(club,week+7,week+13,'published',snap,snap,now()) returning id into wk2;
    select followup into result from public.recap_engagement_editions where recap_id=wk2;
    if result is null or result->'counts' <> '[1,4]' or result->>'tip' <> 'Reset drill' then raise exception 'Missing follow-up %',result; end if;
    if result#>>'{clinic,title}' is distinct from 'Reset clinic' then raise exception 'Relevant dated clinic was omitted'; end if;
    update public.weekly_recaps set published_at=clock_timestamp() where id=wk2;
    if (select count(*) from public.recap_engagement_editions where followup_source_id=wk1) <> 1 then raise exception 'Duplicate result'; end if;
    begin
        perform public.record_recap_engagement(club,week,gen_random_uuid(),'website',fingerprint,'vote','Which drill?','Drops','','');
        raise exception 'closed poll accepted a vote';
    exception when raise_exception then
        if sqlerrm <> 'Voting is closed' then raise; end if;
    end;
    for i in 1..4 loop
        perform public.record_recap_engagement(club,week+7,gen_random_uuid(),'website',fingerprint,'vote','Which drill?','Drops','','');
    end loop;
    insert into public.weekly_recaps(club_id,week_start,week_end,status,final_json,published_json,published_at)
      values(club,week+14,week+20,'published',snap,snap,now()) returning id into wk3;
    if (select followup from public.recap_engagement_editions where recap_id=wk3) is not null then raise exception 'Four votes were published'; end if;
    for i in 1..6 loop
        perform public.record_recap_engagement(club,week+14,gen_random_uuid(),'website',fingerprint,'vote','Which drill?',case when i%2=0 then 'Drops' else 'Resets' end,'','');
    end loop;
    insert into public.weekly_recaps(club_id,week_start,week_end,status,final_json,published_json,published_at)
      values(club,week+21,week+27,'published',snap,snap,now()) returning id into wk4;
    select followup into result from public.recap_engagement_editions where recap_id=wk4;
    if result is null or result->>'winner' is not null or result->'counts' <> '[3,3]' then raise exception 'Tie incorrectly reported'; end if;
    update public.weekly_recaps set status='draft' where id=wk2;
    begin
        perform public.record_recap_engagement(club,week+7,reader_id,'email',fingerprint,'visit','','','','');
        raise exception 'unpublished edition accepted a visit';
    exception when no_data_found then null;
    end;
end $$;
rollback;

-- Run only in an isolated test database after the base queue and lease migrations.
begin;
do $test$
declare
  first_job uuid := gen_random_uuid(); second_job uuid := gen_random_uuid();
  legacy_job uuid := gen_random_uuid(); claimed public.badge_eval_queue;
  claimed_second public.badge_eval_queue; n integer;
begin
  insert into public.badge_eval_queue(id,club_id,event_type,status,created_at)
  values (first_job,'lease_test','match_recorded','pending',now()-interval '2 minutes'),
         (second_job,'lease_test','match_recorded','pending',now()-interval '1 minute'),
         (legacy_job,'lease_test','match_recorded','processing',now()-interval '1 day');

  select * into claimed from public.claim_badge_eval_queue_job_leased('lease_test','worker-a',120);
  assert claimed.id = first_job, 'claims oldest pending job';
  assert claimed.claim_token is not null and claimed.worker_id = 'worker-a';
  assert claimed.started_at is not null and claimed.heartbeat_at is not null;
  assert claimed.attempts = 1 and claimed.lease_expires_at > now();
  select count(*) into n from public.claim_badge_eval_queue_job_leased('lease_test','worker-b',120);
  assert n = 0, 'only one active leased job per club';
  assert not public.heartbeat_badge_eval_queue_job(first_job,gen_random_uuid(),120), 'stale heartbeat refused';
  assert not public.finish_badge_eval_queue_job(first_job,gen_random_uuid(),'done'), 'stale completion refused';
  assert public.heartbeat_badge_eval_queue_job(first_job,claimed.claim_token,120);

  update public.badge_eval_queue set lease_expires_at = now()-interval '1 second' where id=first_job;
  assert not public.heartbeat_badge_eval_queue_job(first_job,claimed.claim_token,120), 'expired lease cannot renew';
  assert not public.finish_badge_eval_queue_job(first_job,claimed.claim_token,'done'), 'expired lease cannot finish';
  select * into claimed_second from public.claim_badge_eval_queue_job_leased('lease_test','worker-b',120);
  assert claimed_second.id = second_job, 'expired processing job is never replayed';
  assert public.finish_badge_eval_queue_job(second_job,claimed_second.claim_token,'done');
  assert not public.finish_badge_eval_queue_job(second_job,claimed_second.claim_token,'done'), 'completion is fenced once';
  assert (select status='processing' and attempts=1 from public.badge_eval_queue where id=first_job);
  assert (select status='processing' and claim_token is null from public.badge_eval_queue where id=legacy_job), 'legacy job remains untouched';
  assert not has_function_privilege('anon','public.claim_badge_eval_queue_job_leased(text,text,integer)','EXECUTE');
  assert not has_function_privilege('authenticated','public.heartbeat_badge_eval_queue_job(uuid,uuid,integer)','EXECUTE');
  assert not has_function_privilege('anon','public.finish_badge_eval_queue_job(uuid,uuid,text,text)','EXECUTE');
  assert has_function_privilege('service_role','public.finish_badge_eval_queue_job(uuid,uuid,text,text)','EXECUTE');
end
$test$;
rollback;

// Retired badges still complete the existing correction ledger without award writes.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
assert.ok(process.env.PGLITE_MODULE, 'Set PGLITE_MODULE to an installed PGlite module');
const { PGlite } = await import(process.env.PGLITE_MODULE);
const db = new PGlite();
const sql = await readFile(new URL('../../supabase/migrations/20260726143742_match_exclusion_atomic_recovery.sql', import.meta.url), 'utf8');
function definition(pattern) { const m = sql.match(pattern); assert.ok(m); return m[0]; }
await db.exec(`
create table public.replay_jobs(id uuid primary key);
create table public.player_badges(id uuid primary key default gen_random_uuid(), club_id text, player_id bigint, badge_id text, earned_at timestamptz, context_type text, context_id text, match_id text, value_num numeric, value_json jsonb, awarded_by text, rule_version text, revoked_at timestamptz, revoked_by uuid, revoke_reason text);
create table public.admin_activity_log(club_id text, actor_email text, actor_role text, action_type text, entity_type text, entity_id text, before_json jsonb, after_json jsonb, note text, source_page text, flagged_for_review boolean);
`);
for (const table of ['match_exclusion_operations', 'match_exclusion_badge_progress']) {
  await db.exec(definition(new RegExp(`create table if not exists public\\.${table} \\([\\s\\S]*?\\n\\);`)));
}
for (const name of ['claim_match_exclusion_badge_progress', 'finalize_match_exclusion_operation']) {
  await db.exec(definition(new RegExp(`create or replace function public\\.${name}\\([\\s\\S]*?\\$function\\$;`)));
}
for (const version of ['v1', 'v2']) {
  const contract = `jupr:match-exclusion-badges:${version}`;
  const club = `retirement-${version}`;
  const id = (await db.query('select gen_random_uuid() as id')).rows[0].id;
  await db.query('insert into replay_jobs values($1)', [id]);
  await db.query(`insert into match_exclusion_operations(id,club_id,mode,idempotency_key,status,targets_json,before_json,after_json,excluded_match_ids,affected_player_ids,badge_ids,badge_contract_version,replay_target,replay_job_id,actor_email,actor_role,source,delete_note) values($1::uuid,$2,'exclude',$1::text,'pending_badge_reconcile','[{}]','[{}]','[{}]',array[1],array[7],array['first_win'],$3,'ALL (Full System Reset)',$1,'test@example.test','admin','test','test')`, [id,club,contract]);
  await db.query('insert into match_exclusion_badge_progress(operation_id,club_id,player_id,badge_ids,badge_contract_version) values($1,$2,7,array[\'first_win\'],$3)', [id,club,contract]);
  await db.query(`insert into player_badges(club_id,player_id,badge_id,context_type,context_id,awarded_by,revoked_at) values($1,7,'first_win','overall','engine-award','engine',null),($1,7,'first_win','overall','manual-award','manual',null),($1,7,'first_win','overall','old-revocation','engine',now()),($1,7,'tournament_champion','overall','trophy','engine',null)`, [club]);
  const before = (await db.query('select * from player_badges where club_id=$1 order by id', [club])).rows;
  const claim = async () => (await db.query(`select claim_match_exclusion_badge_progress($1,$2,'retirement-worker',120,true) as result`, [id,club])).rows[0].result;
  let owned = await claim();
  const ack = async (token=owned.lease_token, worker='retirement-worker', scopedClub=club) => db.query(`
    update match_exclusion_badge_progress set status='succeeded', result_json=$1::jsonb,
      error_text=null, lease_token=null, leased_by=null, lease_expires_at=null, heartbeat_at=null,
      finished_at=now(), updated_at=now()
    where id=$2 and operation_id=$3 and club_id=$4 and player_id=7
      and status='running' and lease_token=$5 and leased_by=$6 and lease_expires_at > 'now'::timestamptz
    returning id`, [JSON.stringify({ok:true,status:'succeeded',skipped_reason:'badges_retired',inserted_count:0,updated_count:0,revoked_count:0}),owned.progress_id,id,scopedClub,token,worker]);
  assert.equal((await ack('00000000-0000-0000-0000-000000000000')).rows.length,0);
  assert.equal((await ack(owned.lease_token,'wrong-worker')).rows.length,0);
  assert.equal((await ack(owned.lease_token,'retirement-worker','wrong-club')).rows.length,0);
  await db.query(`update match_exclusion_badge_progress set lease_expires_at=now()-interval '1 second' where id=$1`, [owned.progress_id]);
  assert.equal((await ack()).rows.length,0);
  const oldToken = owned.lease_token;
  owned = await claim();
  assert.notEqual(owned.lease_token,oldToken);
  assert.equal((await ack(oldToken)).rows.length,0);
  assert.equal((await ack()).rows.length,1);
  const finalize = async () => (await db.query(`select finalize_match_exclusion_operation($1,$2,'test@example.test','admin','test') as result`, [id,club])).rows[0].result;
  const completed = await finalize();
  assert.equal(completed.operation_status,'succeeded');
  assert.equal(completed.badge_progress_total,1);
  assert.equal(completed.inserted_count + completed.updated_count + completed.revoked_count,0);
  assert.equal(completed.badge_results[0].skipped_reason,'badges_retired');
  assert.equal((await finalize()).idempotent,true);
  assert.deepEqual((await db.query('select * from player_badges where club_id=$1 order by id',[club])).rows,before);
}
await db.close();
console.log('PASS: retired correction progress preserves every award, fences token/worker/club/expiry, and finalizes v1/v2 recovery idempotently.');

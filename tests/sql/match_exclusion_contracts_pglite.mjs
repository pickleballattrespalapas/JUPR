// Real PostgreSQL routines, synthetic data only; no network or production writes.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
assert.ok(process.env.PGLITE_MODULE, 'Set PGLITE_MODULE to an installed PGlite module');
const { PGlite } = await import(process.env.PGLITE_MODULE);
const db = new PGlite();
const sql = await readFile(new URL('../../supabase/migrations/20260726143742_match_exclusion_atomic_recovery.sql', import.meta.url), 'utf8');
function definition(pattern) { const m = sql.match(pattern); assert.ok(m); return m[0]; }
await db.exec(`create table public.replay_jobs(id uuid primary key);
create table public.player_badges(id uuid primary key default gen_random_uuid(), club_id text, player_id bigint, badge_id text, earned_at timestamptz, context_type text, context_id text, match_id text, value_num numeric, value_json jsonb, awarded_by text, rule_version text, revoked_at timestamptz, revoked_by uuid, revoke_reason text);`);
for (const table of ['match_exclusion_operations', 'match_exclusion_badge_progress']) {
  await db.exec(definition(new RegExp(`create table if not exists public\\.${table} \\([\\s\\S]*?\\n\\);`)));
}
for (const name of ['claim_match_exclusion_badge_progress', 'apply_match_exclusion_badge_reconciliation']) {
  await db.exec(definition(new RegExp(`create or replace function public\\.${name}\\([\\s\\S]*?\\$function\\$;`)));
}
for (const [version, badge] of [['v1', 'first_win'], ['v2', 'above_expectations']]) {
  const contract = `jupr:match-exclusion-badges:${version}`;
  const club = `synthetic-${version}`;
  const id = (await db.query('select gen_random_uuid() as id')).rows[0].id;
  await db.query('insert into replay_jobs values($1)', [id]);
  await db.query(`insert into match_exclusion_operations(id,club_id,mode,idempotency_key,status,targets_json,before_json,after_json,excluded_match_ids,affected_player_ids,badge_ids,badge_contract_version,replay_target,replay_job_id,actor_email,actor_role,source,delete_note) values($1::uuid,$2,'exclude',$1::text,'pending_badge_reconcile','[{}]','[{}]','[{}]',array[1],array[7],array[$3],$4,'ALL (Full System Reset)',$1,'test@example.test','admin','test','test')`, [id,club,badge,contract]);
  await db.query('insert into match_exclusion_badge_progress(operation_id,club_id,player_id,badge_ids,badge_contract_version) values($1,$2,7,array[$3],$4)', [id,club,badge,contract]);
  await db.query(`insert into player_badges(club_id,player_id,badge_id,context_type,context_id,awarded_by) values($1,7,'outside_contract','overall','protected','engine'),($1,7,$2,'overall','manual','manual')`, [club,badge]);
  const claim = (await db.query(`select claim_match_exclusion_badge_progress($1,$2,'test-worker',120,true) as result`, [id,club])).rows[0].result;
  assert.equal(claim.claimed,true); assert.equal(claim.badge_contract_version,contract);
  const desired = [{badge_id:badge,context_type:'overall',context_id:'synthetic-award',rule_version:contract}];
  const apply = rows => db.query(`select apply_match_exclusion_badge_reconciliation($1,$2,$3,$4,'test-worker',$5::jsonb,'test@example.test') as result`, [id,claim.progress_id,club,claim.lease_token,JSON.stringify(rows)]);
  await assert.rejects(apply([{...desired[0],rule_version:'wrong-version'}]), /DESIRED_INVALID/);
  await assert.rejects(apply([{...desired[0],badge_id:'outside_contract'}]), /BADGE_NOT_ALLOWED/);
  const result=(await apply(desired)).rows[0].result;
  assert.equal(result.status,'succeeded'); assert.equal(result.badge_contract_version,contract); assert.equal(result.inserted_count,1);
  assert.equal((await apply(desired)).rows[0].result.idempotent,true);
  const rows=(await db.query('select badge_id,context_id,rule_version,revoked_at from player_badges where club_id=$1',[club])).rows;
  assert.equal(rows.find(r=>r.context_id==='synthetic-award').rule_version,contract);
  assert.equal(rows.find(r=>r.context_id==='protected').revoked_at,null);
  assert.equal(rows.find(r=>r.context_id==='manual').revoked_at,null);
}
await db.close();
console.log('PASS: real claim/apply SQL accepts v1 and v2, enforces saved version/scope, preserves unrelated/manual badges, and retries idempotently.');

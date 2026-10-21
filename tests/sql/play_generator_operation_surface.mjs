import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const migrations = path.join(root, 'supabase/migrations');
const migrationFile = fs.readdirSync(migrations).find(name => /^\d+_play_generator_operation_surface\.sql$/.test(name));
assert(migrationFile, 'Generator operation surface migration must exist');
const migration = fs.readFileSync(path.join(migrations, migrationFile), 'utf8');
const db = new PGlite();
await db.exec(`
  create role anon; create role authenticated; create role service_role bypassrls;
  create table public.matches (club_id text, context_type text, context_id text);
`);
await db.exec(fs.readFileSync(path.join(migrations, '20260719201500_live_ladder_admin_operations.sql'), 'utf8'));
await db.exec('set role service_role');
let sequence = 0;
async function insert(surface, overrides = {}) {
  const n = ++sequence;
  const values = { operation_key: String(n).padStart(64,'0'), club_id:'club', surface,
    operation_type:'start_session', entity_id:'new', idempotency_key:`generator-key-${n}`,
    request_fingerprint:'a'.repeat(64), expected_version:'new', status:'intent',
    created_by:'operator@example.test', updated_by:'operator@example.test', ...overrides };
  const columns = Object.keys(values);
  return db.query(`insert into public.live_ladder_admin_operations (${columns.join(',')}) values (${columns.map((_,i)=>`$${i+1}`).join(',')}) returning *`, Object.values(values));
}
await assert.rejects(()=>insert('play_generator'), error => error.code==='23514' && error.constraint==='live_ladder_admin_operations_surface_check');
assert.equal((await db.query('select count(*) n from live_ladder_admin_operations')).rows[0].n,0);
await db.exec('reset role');
await db.exec(migration);
await db.exec(migration);
await db.exec('set role service_role');
const first = (await insert('play_generator')).rows[0];
assert.equal(first.status,'intent');
// The active-version lease still blocks concurrent creates from the same snapshot.
await assert.rejects(()=>insert('play_generator'), error => error.code==='23505' && error.constraint==='live_ladder_admin_operations_active_version_unique');
// Completing the first intent permits a separate session, but reusing its key does not.
await db.query(`update live_ladder_admin_operations set status='completed' where operation_key=$1`,[first.operation_key]);
await insert('play_generator');
await assert.rejects(()=>insert('play_generator',{operation_key:first.operation_key}), error => error.code==='23505');
for (const surface of ['challenge_ladder','moneyball','jupr_live_admin']) {
  assert.equal((await insert(surface)).rows[0].surface,surface);
}
await assert.rejects(()=>insert('unknown_surface'), error => error.code==='23514' && error.constraint==='live_ladder_admin_operations_surface_check');
await assert.rejects(()=>insert('moneyball',{expected_version:'different',status:'invalid'}), error => error.code==='23514' && error.constraint==='live_ladder_admin_operations_status_check');
await db.exec('reset role');
const protections = (await db.query(`select relrowsecurity,relforcerowsecurity,
  has_table_privilege('anon','public.live_ladder_admin_operations','insert') anon_write,
  has_table_privilege('authenticated','public.live_ladder_admin_operations','select') authenticated_read
  from pg_class where oid='public.live_ladder_admin_operations'::regclass`)).rows[0];
assert.deepEqual(protections,{relrowsecurity:true,relforcerowsecurity:true,anon_write:false,authenticated_read:false});
console.log('Generator journal SQL regression passed: reproduced old CHECK failure; generator intents now persist; original surfaces, active-version lease, duplicate keys, lifecycle CHECK and private RLS remain enforced.');
await db.close();

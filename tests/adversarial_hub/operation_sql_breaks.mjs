// Execute actual migrations in disposable in-memory PGlite, never a remote DB.
// Acceptance checks for terminal conflicts, atomic receipts and private access.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const db = new PGlite();
const trace = (scenario, data) => console.log('AUDIT_TRACE ' + JSON.stringify({ scenario, ...data }));
try {
  await db.exec(`
    create role anon; create role authenticated; create role service_role bypassrls;
    create table public.matches (club_id text, context_type text, context_id text);
    create table public.public_live_operations (operation_key text primary key, club_id text, session_key text, request_fingerprint text, status text);
    grant select,insert,update on public.public_live_operations to service_role;
    create table public.live_sessions (
      id uuid primary key default gen_random_uuid(), club_id text not null, session_key text not null,
      source text, created_by_email text, state jsonb not null default '{}', title text, status text not null default 'active',
      version bigint not null default 1, created_at timestamptz not null default clock_timestamp(),
      updated_at timestamptz not null default clock_timestamp(), last_seen_at timestamptz,
      expires_at timestamptz, completed_at timestamptz, last_operation_key text,
      last_request_fingerprint text, pending_operation_key text, unique(club_id,session_key)
    );
    grant select,insert,update,delete on public.live_sessions to service_role;
  `);
  for (const name of [
    '20260719201500_live_ladder_admin_operations.sql',
    '20261020043027_play_generator_operation_surface.sql',
    '20261020033713_round_robin_hub_atomic_operations.sql',
    '20261020161332_round_robin_hub_participant_identities.sql',
    '20261023052802_round_robin_operation_recovery.sql',
    '20261028230610_round_robin_hub_shared_courts.sql',
  ]) await db.exec(fs.readFileSync(path.join(root, 'supabase/migrations', name), 'utf8'));
  await db.exec('set role service_role');
  const hubId = '71000000-0000-4000-8000-000000000001';
  let sequence = 0;
  async function commit(version, state, sessions, key = `audit-hub-${++sequence}`) {
    const result = await db.query('select public.commit_round_robin_hub($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) result',
      ['club-a', hubId, version, key, 'f'.repeat(64), 'Audit Hub', 'active', state, sessions, 'audit@example.test']);
    return result.rows[0].result;
  }
  const state = { courts: [], groups: [] };
  await commit(0, state, []);
  for (const key of ['rr-a', 'rr-b']) {
    await db.query('insert into live_sessions(club_id,session_key,state) values($1,$2,$3)', [
      'club-a', key, { mode: 'admin_play_generator', page_state: { event: {
        generatorKind: 'round_robin', participants: [], rounds: [], currentRoundNumber: 1,
      } } },
    ]);
  }
  const rows = (await db.query('select * from live_sessions order by session_key')).rows;
  state.groups = rows.map(row => ({ session_key: row.session_key }));
  const attached = await commit(1, state, rows.map(row => ({
    session_key: row.session_key, expected_version: row.updated_at, expected_numeric_version: row.version,
    patch: { state: { ...row.state, round_robin_hub_id: hubId } },
  })));
  const currentRows = (await db.query('select * from live_sessions order by session_key')).rows;
  const version = currentRows[0].updated_at;
  assert.equal((await db.query('select count(distinct updated_at) n from live_sessions')).rows[0].n, 1);
  assert.equal(currentRows[1].updated_at.getTime(), version.getTime());
  trace('sql-hub-gives-sibling-sessions-identical-version', {
    entities: currentRows.map(row => row.session_key), shared_timestamp: version,
  });
  async function insertOperation(entity, key, status) {
    return db.query(`insert into live_ladder_admin_operations
      (operation_key,club_id,surface,operation_type,entity_id,idempotency_key,request_fingerprint,
       expected_version,status,created_by,updated_by)
      values($1,'club-a','play_generator','save_round_scores',$2,$3,$4,$5,$6,'audit@example.test','audit@example.test')`,
    [key.padStart(64, '0'), entity, `audit-key-${key}`, 'e'.repeat(64), version.toISOString(), status]);
  }
  await insertOperation('rr-a', '1', 'recovery_required');
  await insertOperation('rr-b', '2', 'intent');
  let blocked;
  try { await insertOperation('rr-a', '3', 'intent'); }
  catch (error) { blocked = error; }
  assert.equal(blocked?.code, '23505');
  assert.equal(blocked?.constraint, 'live_ladder_admin_operations_generator_version_unique');
  trace('sql-entity-scoped-lease', { sibling_session_allowed: true, same_session_blocked: true });
  await db.query("update live_ladder_admin_operations set status='failed' where entity_id='rr-a'");
  await insertOperation('rr-a', '3', 'running');
  const readSet = currentRows.map(row => ({ session_key: row.session_key,
    expected_version: row.updated_at, expected_numeric_version: row.version, patch: {} }));
  const noOp = await commit(attached.hub.version, state, readSet);
  assert.equal(noOp.hub.version, attached.hub.version);
  assert.deepEqual((await db.query('select * from live_sessions order by session_key')).rows, currentRows);
  trace('sql-empty-session-patches-preserve-hub-version', {
    before: attached.hub.version, after: noOp.hub.version, session_writes: 0,
  });
  let stale;
  try { await commit(attached.hub.version - 1, state, readSet, 'audit-stale-key'); }
  catch (error) { stale = error; }
  assert.equal(stale?.code, 'PT409');
  assert.equal(stale?.message, 'The Round Robin Hub changed. Reload before saving.');
  assert.equal((await db.query('select version from round_robin_hubs')).rows[0].version, noOp.hub.version);
  assert.equal((await db.query("select count(*) n from round_robin_hub_operations where operation_key='audit-stale-key'")).rows[0].n, 0);
  trace('sql-stale-version-terminal-conflict', {
    sqlstate: stale.code, state_committed: false, receipt_committed: false,
    limitation: 'PGlite does not execute PostgREST/Hasql retries',
  });
  // This validation happens after session UPDATE and receipt INSERT. Failure
  // must roll back both, plus all court/player leases and the hub receipt.
  const invalidState = structuredClone(state);
  invalidState.courts = [{ id: 'late-failure-court', label: 'Audit Court', open: true,
    current: { session_key: 'rr-a', match_id: 'does-not-exist', round_number: 1 } }];
  const latePatches = structuredClone(readSet);
  latePatches[0].patch = { state: structuredClone(currentRows[0].state),
    last_operation_key: '3'.padStart(64, '0'), last_request_fingerprint: 'e'.repeat(64) };
  latePatches[0].patch.state.page_state.event.name = 'This update must roll back';
  let lateFailure;
  try { await commit(noOp.hub.version, invalidState, latePatches, 'audit-late-rollback'); }
  catch (error) { lateFailure = error; }
  assert.equal(lateFailure?.code, '22023');
  assert.match(lateFailure.message, /assigned match is no longer present/);
  assert.deepEqual((await db.query('select * from live_sessions order by session_key')).rows, currentRows);
  assert.equal((await db.query('select version from round_robin_hubs')).rows[0].version, noOp.hub.version);
  assert.equal((await db.query("select count(*) n from round_robin_hub_operations where operation_key='audit-late-rollback'")).rows[0].n, 0);
  assert.equal((await db.query('select count(*) n from round_robin_hub_court_leases')).rows[0].n, 0);
  assert.equal((await db.query('select count(*) n from play_generator_mutation_receipts')).rows[0].n, 0);
  trace('sql-late-validation-atomic-rollback-control', {
    sqlstate: lateFailure.code, earlier_session_update_rolled_back: true,
    court_lease_committed: false, hub_version_changed: false, receipt_committed: false,
  });
  const linkedPatches = structuredClone(readSet);
  linkedPatches[0].patch = { state: currentRows[0].state, last_operation_key: '3'.padStart(64, '0'), last_request_fingerprint: 'e'.repeat(64) };
  await commit(noOp.hub.version, state, linkedPatches, 'audit-atomic-receipt');
  assert.equal((await db.query('select count(*) n from play_generator_mutation_receipts')).rows[0].n, 1);
  trace('sql-hub-receipt-bridge', { atomic_receipts: 1 });

  for (const [key, mode, kind] of [['standalone-admin', 'admin_play_generator', 'admin'], ['standalone-public', 'public_play_generator', 'public'], ['standalone-shared', 'admin_play_generator', 'public']]) {
    await db.query('insert into live_sessions(club_id,session_key,state) values($1,$2,$3)', ['club-a', key, {mode, page_state:{event:{generatorKind:'round_robin', name:'before'}}}]);
    const before = (await db.query('select * from live_sessions where session_key=$1', [key])).rows[0];
    const opKey = key.padStart(64, '0');
    if (kind === 'admin') await insertOperation(key, key, 'running');
    else await db.query("insert into public_live_operations values($1,'club-a',$2,$3,'intent')", [opKey,key,'e'.repeat(64)]);
    const patch = {state:{...before.state, page_state:{event:{generatorKind:'round_robin', name:'saved once'}}}};
    const args = ['club-a', key, before.updated_at, before.version, patch, opKey, 'e'.repeat(64)];
    const sql = `select commit_${kind}_play_generator_session($1,$2,$3,$4,$5,$6,$7) result`;
    const committed = (await db.query(sql,args)).rows[0].result;
    assert.equal(committed.version, before.version+1);
    await db.query("update live_sessions set title='later edit',last_operation_key='another-operation',version=version+1 where session_key=$1", [key]);
    const replay = (await db.query(sql,args)).rows[0].result;
    assert.equal(replay.version, committed.version+1);
    assert.equal(replay.title, 'later edit');
    assert.equal((await db.query('select count(*) n from play_generator_mutation_receipts where operation_key=$1',[opKey])).rows[0].n, 1);
    let mismatch;
    try { await db.query(sql,[...args.slice(0,6), 'x'.repeat(64)]); } catch (error) { mismatch=error; }
    assert.equal(mismatch?.code,'22023');
    trace('sql-durable-receipt-survives-later-edit', {surface:kind, mode, version:replay.version, duplicate_writes:0});
  }
  const newKey='new-admin-operation'.padStart(64,'0');
  await insertOperation('new','new-admin-operation','running');
  const created=(await db.query('select commit_admin_play_generator_session($1,$2,null,null,$3,$4,$5) result',
    ['club-a','created-admin',{club_id:'club-a',session_key:'created-admin',title:'New',state:{mode:'admin_play_generator'}},newKey,'e'.repeat(64)])).rows[0].result;
  assert.equal(created.session_key,'created-admin');
  assert.equal((await db.query('select count(*) n from play_generator_mutation_receipts where operation_key=$1',[newKey])).rows[0].n,1);

  // Keep one real assignment when merging an idle duplicate; reject a merge
  // that would combine two previously distinct court commitments.
  const identityHub='71000000-0000-4000-8000-000000000002';
  const identityState={identity_version:2,groups:[{session_key:'identity-a'},{session_key:'identity-b'}],
    courts:[{id:'identity-c1',label:'Identity One',open:true,current:{session_key:'identity-a',match_id:'m1',round_number:1},next:null},
      {id:'identity-c2',label:'Identity Two',open:true,current:null,next:{session_key:'identity-b',match_id:'m1',round_number:1}}],
    identity_decisions:[{left:{session_key:'identity-a',participant_id:'p1'},right:{session_key:'identity-b',participant_id:'p1'},resolution:'different'}],
    participant_identities:{}};
  for (const key of ['identity-a','identity-b']) {
    identityState.participant_identities[key]={p1:`guest:${key}:p1`,p2:`guest:${key}:p2`};
    await db.query('insert into live_sessions(club_id,session_key,state) values($1,$2,$3)', ['club-a',key,
      {mode:'admin_play_generator',page_state:{event:{generatorKind:'round_robin',participants:[{id:'p1',name:'Alex'},{id:'p2',name:key+' Other'}],
      rounds:[{number:1,status:'active',matches:[{id:'m1',sideA:['p1'],sideB:['p2']}]}]}}}]);
  }
  let identityVersion=0;
  async function identityCommit(value) {
    const members=(await db.query("select * from live_sessions where session_key in ('identity-a','identity-b') order by session_key")).rows;
    const reads=members.map(row=>({session_key:row.session_key,expected_version:row.updated_at,expected_numeric_version:row.version,
      patch:row.state.round_robin_hub_id ? {} : {state:{...row.state,round_robin_hub_id:identityHub}}}));
    const result=(await db.query('select commit_round_robin_hub($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) result',
      ['club-a',identityHub,identityVersion,`identity-recovery-${++sequence}`,'f'.repeat(64),'Identity merge','active',value,reads,'audit@example.test'])).rows[0].result;
    identityVersion=result.hub.version;
    return result;
  }
  await identityCommit(identityState);
  const mergedIdentity=structuredClone(identityState);
  mergedIdentity.identity_decisions[0].resolution='same';
  mergedIdentity.participant_identities['identity-b'].p1='guest:identity-a:p1';
  await assert.rejects(identityCommit(mergedIdentity), error=>error.code==='22023' && /two committed/.test(error.message));
  identityState.courts[1].next=null;
  await identityCommit(identityState);
  mergedIdentity.courts[1].next=null;
  const merged=await identityCommit(mergedIdentity);
  assert.deepEqual(merged.hub.state.courts[0].current,identityState.courts[0].current);
  assert.equal((await db.query('select count(*) n from round_robin_hub_player_leases where hub_id=$1',[identityHub])).rows[0].n,2);
  trace('sql-identity-merge-keeps-one-commitment', {one_busy_identity_allowed:true,two_busy_identities_rejected:true,original_game_preserved:true});

  for (const role of ['anon','authenticated']) {
    await db.exec(`reset role;set role ${role}`);
    await assert.rejects(db.query('select * from play_generator_mutation_receipts'), error => error.code==='42501');
    await assert.rejects(db.query('select commit_admin_play_generator_session(null,null,null,null,null,null,null)'), error=>error.code==='42501');
    await assert.rejects(db.query('select commit_public_play_generator_session(null,null,null,null,null,null,null)'), error=>error.code==='42501');
  }
  trace('sql-private-receipt-boundary', {anon_denied:true, authenticated_denied:true});
} finally {
  await db.close();
}

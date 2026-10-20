// Run against an isolated, in-memory PostgreSQL engine:
// NODE_PATH=<directory containing @electric-sql/pglite> node tests/team-match-rpc-permissions.cjs
// This test never connects to Supabase or reads application credentials.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { PGlite } = require('@electric-sql/pglite');

const migration = fs.readFileSync(path.join(__dirname, '..', 'supabase', 'migrations',
  '20261109004105_restrict_team_match_admin_rpc_execution.sql'), 'utf8');
const routines = [
  ['admin_replace_team_match_schedule_v1', 'text,text,integer,jsonb,text'],
  ['admin_save_team_match_team_v1', 'text,text,text,text,text,jsonb,integer,integer,text'],
  ['admin_update_team_match_competition_v1', 'text,text,integer,jsonb,text'],
];
const signature = ([name, args]) => `public.${name}(${args})`;
const call = ([name, args]) => `select public.${name}(${args.split(',').map(type => `null::${type}`).join(',')}) as result`;

async function snapshot(db) {
  return (await db.query(`
    select p.oid, p.oid::regprocedure::text as signature, p.proowner, p.prosecdef,
           p.proconfig, pg_get_functiondef(p.oid) as definition
      from pg_proc p join pg_namespace n on n.oid = p.pronamespace
     where n.nspname = 'public'
     order by p.oid
  `)).rows;
}

async function asRole(db, role, sql) {
  await db.exec(`set role ${role}`);
  try {
    return await db.query(sql);
  } finally {
    await db.exec('reset role');
  }
}

async function run() {
  const db = new PGlite();
  try {
    await db.exec(`create role anon; create role authenticated; create role service_role;
      create role existing_backend; grant usage on schema public to public;`);

    // A deployment without the legacy feature must remain valid.
    await db.exec(migration);
    assert.equal((await db.query(`select count(*)::int as count from pg_proc p
      join pg_namespace n on n.oid = p.pronamespace where n.nspname = 'public'`)).rows[0].count, 0);

    for (const routine of routines) {
      await db.exec(`create function ${signature(routine)} returns jsonb
        language sql security definer set search_path = ''
        as $$select '{"ok":true}'::jsonb$$;
        grant execute on function ${signature(routine)} to service_role, existing_backend;`);
    }
    // Reproduce both inherited PUBLIC access and explicit role grants.
    await db.exec(`grant execute on function ${signature(routines[1])} to anon;
      grant execute on function ${signature(routines[2])} to authenticated;
      create function public.admin_save_team_match_team_v1() returns jsonb
        language sql as $$select '{"unrelated":true}'::jsonb$$;`);
    for (const routine of routines) {
      for (const role of ['anon', 'authenticated']) {
        assert.deepEqual((await asRole(db, role, call(routine))).rows[0].result, { ok: true });
      }
    }

    const before = await snapshot(db);
    await db.exec(migration);
    await db.exec(migration); // Reapplication must retain the same boundary.
    assert.deepEqual(await snapshot(db), before, 'Bodies, signatures, owners and security settings must be unchanged');

    for (const routine of routines) {
      const privileges = (await db.query(`select
        has_function_privilege('anon', $1, 'EXECUTE') as anon,
        has_function_privilege('authenticated', $1, 'EXECUTE') as authenticated,
        has_function_privilege('service_role', $1, 'EXECUTE') as service_role,
        has_function_privilege('existing_backend', $1, 'EXECUTE') as existing_backend`,
        [signature(routine)])).rows[0];
      assert.deepEqual(privileges, { anon: false, authenticated: false, service_role: true, existing_backend: true });
      for (const role of ['anon', 'authenticated']) {
        await assert.rejects(asRole(db, role, call(routine)), error => error.code === '42501');
      }
      assert.deepEqual((await asRole(db, 'service_role', call(routine))).rows[0].result, { ok: true });
      assert.deepEqual((await asRole(db, 'existing_backend', call(routine))).rows[0].result, { ok: true });
    }
    assert.deepEqual((await asRole(db, 'anon', 'select public.admin_save_team_match_team_v1() as result')).rows[0].result,
      { unrelated: true }, 'A different overload must retain its existing permissions');

    // Partial legacy deployments are safe too; no absent routine is recreated.
    await db.exec(`drop function ${signature(routines[0])};`);
    await db.exec(migration);
    assert.equal((await db.query('select to_regprocedure($1) as routine', [signature(routines[0])])).rows[0].routine, null);
    console.log('Team-match RPC permissions: inherited/direct grants closed; service role, unrelated grants, bodies and overload preserved; absent/partial/repeated migration passed.');
  } finally {
    await db.close();
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });

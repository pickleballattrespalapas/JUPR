import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
// An in-memory PostgreSQL instance only: never connects to hosted Supabase.
const db = new PGlite();
try {
  await db.exec('create role anon; create role authenticated; create role service_role bypassrls;');
  for (const name of [
    'supabase/migrations/20260718141016_badge_eval_queue_atomic_club_claim.sql',
    'supabase/migrations/20261108234006_badge_queue_worker_leases.sql',
    'tests/sql/badge_queue_worker_leases.sql',
  ]) {
    await db.exec(fs.readFileSync(path.join(root, name), 'utf8'));
  }
  console.log('Badge worker lease SQL passed: claim ownership, heartbeat, expiry, fenced completion, no replay, and privileges.');
} finally {
  await db.close();
}

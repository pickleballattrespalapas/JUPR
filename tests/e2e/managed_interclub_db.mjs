// Isolated, loopback-only Postgres for the browser journey. Executes the actual migration/RPC.
import http from 'node:http';
import fs from 'node:fs';
import { PGlite } from '../sql/node_modules/@electric-sql/pglite/dist/index.js';
const db = new PGlite();
await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
 create table clubs(id text primary key);
 insert into clubs values('tres_palapas');
 create table admin_role_assignments(club_id text,email text,user_id uuid,role text,revoked_at timestamptz,expires_at timestamptz);
 insert into admin_role_assignments values('tres_palapas','joe@example.invalid','10000000-0000-4000-8000-000000000001','administrator',null,null);
 create table pcs_interclub_seasons(id uuid primary key,organizer_club_id text);
 grant select on admin_role_assignments,clubs,pcs_interclub_seasons to service_role;`);
await db.exec(fs.readFileSync(new URL('../../supabase/migrations/20261114223140_interclub_organizer_managed.sql', import.meta.url), 'utf8'));
await db.exec('set role service_role');
http.createServer(async (req, res) => {
 res.setHeader('Content-Type', 'application/json');
 if (req.method === 'GET') { res.end('{"ready":true}'); return; }
 try {
  let body = ''; for await (const chunk of req) body += chunk;
  const { sql, params } = JSON.parse(body);
  const result = await db.query(sql, params || []);
  res.end(JSON.stringify({ data: result.rows }));
 } catch (e) { res.statusCode = 400; res.end(JSON.stringify({ code: e.code, message: e.message })); }
}).listen(8144, '127.0.0.1');

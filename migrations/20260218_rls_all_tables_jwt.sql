do $$
declare
    t record;
    p record;
begin
    for t in
        select c.table_schema, c.table_name
        from information_schema.columns c
        join pg_class pc
          on pc.relname = c.table_name
        join pg_namespace pn
          on pn.oid = pc.relnamespace
         and pn.nspname = c.table_schema
        where c.table_schema = 'public'
          and c.column_name = 'club_id'
          and c.table_name <> 'badges'
          and pc.relkind in ('r', 'p')
        order by c.table_name
    loop
        execute format(
            'alter table %I.%I enable row level security',
            t.table_schema,
            t.table_name
        );

        for p in
            select policyname
            from pg_policies
            where schemaname = t.table_schema
              and tablename = t.table_name
            order by policyname
        loop
            execute format(
                'drop policy if exists %I on %I.%I',
                p.policyname,
                t.table_schema,
                t.table_name
            );
        end loop;

        execute format(
            'create policy %I on %I.%I for select using (club_id = (public.jwt_claims() ->> ''club_id''))',
            t.table_name || '_select_by_club',
            t.table_schema,
            t.table_name
        );

        execute format(
            'create policy %I on %I.%I for insert with check (club_id = (public.jwt_claims() ->> ''club_id''))',
            t.table_name || '_insert_by_club',
            t.table_schema,
            t.table_name
        );

        execute format(
            'create policy %I on %I.%I for update using (club_id = (public.jwt_claims() ->> ''club_id'')) with check (club_id = (public.jwt_claims() ->> ''club_id''))',
            t.table_name || '_update_by_club',
            t.table_schema,
            t.table_name
        );
    end loop;
end $$;

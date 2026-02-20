-- Expose safe column introspection via SECURITY DEFINER RPC
-- Required because PostgREST does not expose information_schema directly.

create or replace function public.get_public_table_columns(p_table text)
returns table(column_name text)
language sql
security definer
as $$
    select column_name
    from information_schema.columns
    where table_schema = 'public'
      and table_name = p_table;
$$;

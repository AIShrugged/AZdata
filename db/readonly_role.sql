-- AZdata least-privilege execution role (defense-in-depth behind src/nlsql.py guard_sql).
--
-- The query executor (execute_readonly) SET ROLEs into azdata_ro before running any
-- model-generated SQL. So even if the SQL guard were bypassed, the session can only
-- SELECT the two catalog tables and — being a non-superuser, NOLOGIN role — cannot
-- write, read any other table, or use superuser-only functions (pg_read_file, etc.).
--
-- Apply once per database:  psql -d azdata -f db/readonly_role.sql

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'azdata_ro') THEN
    CREATE ROLE azdata_ro NOLOGIN;
  END IF;
END
$$;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM azdata_ro;
GRANT USAGE ON SCHEMA public TO azdata_ro;
GRANT SELECT ON einvoice, taxpayer TO azdata_ro;

"""
Migration: add employers.is_approved (privacy gate #7).

- New self-registered employers default to NOT approved.
- Existing employers at migration time are grandfathered (approved) so they keep access.
- Idempotent: re-running will NOT re-approve employers an admin later revoked
  (the grandfather UPDATE only touches rows that are still NULL).

Run from the project root:  python -m Migrations.add_employer_approval
"""
import os
import psycopg2

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL is not set. Define it in .env before running this migration.")


def run_migration():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # 1. Add the column with NO default so pre-existing rows become NULL (distinguishable).
    cur.execute("ALTER TABLE employers ADD COLUMN IF NOT EXISTS is_approved BOOLEAN;")

    # 2. Grandfather only the rows that existed before this column (still NULL).
    cur.execute("UPDATE employers SET is_approved = TRUE WHERE is_approved IS NULL;")

    # 3. New inserts default to FALSE (pending approval) from now on.
    cur.execute("ALTER TABLE employers ALTER COLUMN is_approved SET DEFAULT FALSE;")

    conn.commit()
    cur.close()
    conn.close()
    print("Migration complete: employers.is_approved added; existing employers grandfathered (approved).")


if __name__ == '__main__':
    run_migration()

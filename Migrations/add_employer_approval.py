"""
Migration: add employers.is_approved (privacy gate #7).

- New self-registered employers default to NOT approved.
- Existing employers at migration time are grandfathered (approved) so they keep access.
- Idempotent: re-running will NOT re-approve employers an admin later revoked
  (the grandfather UPDATE only touches rows that are still NULL).

Run from the project root:  python -m Migrations.add_employer_approval
"""
import os
import sys

# Add parent directory to path to find 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.extensions import db
from sqlalchemy import text

app = create_app()

def run_migration():
    with app.app_context():
        # 1. Add the column with NO default so pre-existing rows become NULL (distinguishable).
        db.session.execute(text("ALTER TABLE employers ADD COLUMN IF NOT EXISTS is_approved BOOLEAN;"))

        # 2. Grandfather only the rows that existed before this column (still NULL).
        db.session.execute(text("UPDATE employers SET is_approved = TRUE WHERE is_approved IS NULL;"))

        # 3. New inserts default to FALSE (pending approval) from now on.
        db.session.execute(text("ALTER TABLE employers ALTER COLUMN is_approved SET DEFAULT FALSE;"))

        db.session.commit()
        print("Migration complete: employers.is_approved added; existing employers grandfathered (approved).")

if __name__ == '__main__':
    run_migration()

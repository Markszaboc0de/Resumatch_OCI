import os
import psycopg2

DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    "postgresql://app_user:Mindenszarhoz@localhost:5432/job_match_db"
)

def run_migration():
    print(f"Connecting to database to run migration on 'job_descriptions'...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Add sync_active column
        cur.execute("""
        ALTER TABLE job_descriptions 
        ADD COLUMN IF NOT EXISTS sync_active BOOLEAN DEFAULT FALSE;
        """)
        
        # Add is_direct_upload column
        cur.execute("""
        ALTER TABLE job_descriptions 
        ADD COLUMN IF NOT EXISTS is_direct_upload BOOLEAN DEFAULT TRUE;
        """)
        
        # Update existing records to TRUE so they are preserved
        cur.execute("""
        UPDATE job_descriptions SET is_direct_upload = TRUE WHERE is_direct_upload IS NULL;
        UPDATE job_descriptions SET sync_active = FALSE WHERE sync_active IS NULL;
        """)

        conn.commit()
        cur.close()
        conn.close()
        
        print("Migration successful: added 'sync_active' and 'is_direct_upload' columns.")
    except Exception as e:
        print(f"Migration error: {e}")

if __name__ == '__main__':
    run_migration()

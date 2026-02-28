from app import app, db
from sqlalchemy import text, create_engine

def update_job_schema():
    with app.app_context():
        # Override engine just for schema update
        engine = create_engine("postgresql://postgres:Mindenszarhoz@localhost:5432/job_match_db")
        print(f"Connecting to {engine.url}...")
        
        with engine.connect() as conn:
            print("Checking/Adding 'is_native' column to job_descriptions...")
            try:
                conn.execute(text("ALTER TABLE job_descriptions ADD COLUMN is_native BOOLEAN DEFAULT FALSE;"))
                print("Added 'is_native' column.")
            except Exception as e:
                print(f"Column 'is_native' might already exist or error: {e}")
            
            conn.commit()
    print("Schema update complete.")

if __name__ == "__main__":
    update_job_schema()

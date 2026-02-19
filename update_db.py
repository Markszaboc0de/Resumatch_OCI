from app import app, db
from sqlalchemy import text

with app.app_context():
    # 1. Add Employers Table
    # db.create_all() will create Employers if it doesn't exist
    print("Creating tables (if not exist)...")
    db.create_all()
    
    # 2. Add employer_id column to Job_Descriptions if it doesn't exist
    # Inspecting is hard without inspector, so we'll try to add it and catch error if exists
    print("Checking for employer_id column...")
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE job_descriptions ADD COLUMN employer_id INTEGER REFERENCES employers(employer_id)"))
            conn.commit()
            print("Added employer_id column.")
    except Exception as e:
        print(f"Column likely exists or error: {e}")

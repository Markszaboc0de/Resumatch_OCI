from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Adding employer_id to job_descriptions...")
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE job_descriptions ADD COLUMN employer_id INTEGER REFERENCES employers(employer_id)"))
            conn.commit()
            print("Success")
    except Exception as e:
        print(f"Error: {e}")

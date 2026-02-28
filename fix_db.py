from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        db.session.execute(text("ALTER TABLE job_descriptions ADD COLUMN is_native BOOLEAN DEFAULT FALSE;"))
        db.session.commit()
        print("Successfully added 'is_native' column to job_descriptions table.")
    except Exception as e:
        print(f"Error (maybe column already exists?): {e}")

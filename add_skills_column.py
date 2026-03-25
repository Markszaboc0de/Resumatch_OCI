from app import app, db
from sqlalchemy import text

def add_skills_columns():
    with app.app_context():
        print("Adding extracted_skills columns...")
        
        try:
            db.session.execute(text("ALTER TABLE cvs ADD COLUMN IF NOT EXISTS extracted_skills TEXT"))
            print("Added extracted_skills to cvs table.")
        except Exception as e:
            print(f"Error modifying cvs: {e}")
            db.session.rollback()
            
        try:
            db.session.execute(text("ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS extracted_skills TEXT"))
            print("Added extracted_skills to job_descriptions table.")
        except Exception as e:
            print(f"Error modifying job_descriptions: {e}")
            db.session.rollback()
            
        db.session.commit()
        print("Database migration complete!")

if __name__ == "__main__":
    add_skills_columns()

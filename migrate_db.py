from app import app, db
from sqlalchemy import text

def check_and_migrate():
    with app.app_context():
        # Check if URL column exists
        try:
            result = db.session.execute(text("SELECT url FROM job LIMIT 1"))
            print("URL column exists.")
        except Exception as e:
            print(f"URL column missing, adding it... Error: {e}")
            db.session.rollback()
            try:
                db.session.execute(text("ALTER TABLE job ADD COLUMN url TEXT"))
                db.session.commit()
                print("Added url column to Job table.")
            except Exception as e2:
                print(f"Failed to add column: {e2}")

if __name__ == "__main__":
    check_and_migrate()

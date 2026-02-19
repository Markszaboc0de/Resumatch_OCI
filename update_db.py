from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Creating tables (Contact_Requests if missing)...")
    db.create_all()
    
    print("Checking for contact_email column in employers...")
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE employers ADD COLUMN contact_email VARCHAR(255)"))
            conn.commit()
            print("Added contact_email column.")
    except Exception as e:
        print(f"Column likely exists or error: {e}")

from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Checking for contact_info column in employers table...")
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE employers ADD COLUMN contact_info TEXT"))
            conn.commit()
            print("Added contact_info column to employers table.")
    except Exception as e:
        print(f"Column likely exists or error: {e}")

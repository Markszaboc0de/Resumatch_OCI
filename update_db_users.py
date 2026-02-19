from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Checking for short_description column...")
    try:
        with db.engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN short_description VARCHAR(500)"))
            conn.commit()
            print("Added short_description column to users table.")
    except Exception as e:
        print(f"Column likely exists or error: {e}")

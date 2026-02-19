from app import app, db
from sqlalchemy import text

with app.app_context():
    print("Checking for is_visible column in users table...")
    try:
        with db.engine.connect() as conn:
            # Default to TRUE (Visible)
            conn.execute(text("ALTER TABLE users ADD COLUMN is_visible BOOLEAN DEFAULT TRUE"))
            conn.commit()
            print("Added is_visible column to users table.")
    except Exception as e:
        print(f"Column likely exists or error: {e}")

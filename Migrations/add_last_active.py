from app import app, db
from sqlalchemy import text
from datetime import datetime

with app.app_context():
    try:
        db.session.execute(text('ALTER TABLE users ADD COLUMN IF NOT EXISTS last_active_date TIMESTAMP'))
        db.session.commit()
        print("Successfully added last_active_date to users.")
        
        # update existing users
        db.session.execute(text("UPDATE users SET last_active_date = :now WHERE last_active_date IS NULL"), {"now": datetime.utcnow()})
        db.session.commit()
        print("Successfully populated last_active_date for existing users.")
    except Exception as e:
        db.session.rollback()
        print(f"Error: {e}")

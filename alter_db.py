from app import app, db, Admins
from sqlalchemy import text
from werkzeug.security import generate_password_hash

with app.app_context():
    # Users
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS match_limit INTEGER DEFAULT 10"))
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS matches_used_this_month INTEGER DEFAULT 0"))
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS match_reset_date TIMESTAMP"))
    
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS supersearch_limit INTEGER DEFAULT 5"))
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS supersearch_used_this_month INTEGER DEFAULT 0"))
    db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS supersearch_reset_date TIMESTAMP"))

    # Employers
    db.session.execute(text("ALTER TABLE employers ADD COLUMN IF NOT EXISTS match_limit INTEGER DEFAULT 10"))
    db.session.execute(text("ALTER TABLE employers ADD COLUMN IF NOT EXISTS matches_used_this_month INTEGER DEFAULT 0"))
    db.session.execute(text("ALTER TABLE employers ADD COLUMN IF NOT EXISTS match_reset_date TIMESTAMP"))
    
    # Admins
    db.create_all()
    
    # Create default admin
    if Admins.query.count() == 0:
        a = Admins(username="admin", password_hash=generate_password_hash("admin123"))
        db.session.add(a)
    
    db.session.commit()
    print("Database updated!")

import os
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
    
    # Create the first admin ONLY from env-provided credentials (no known default).
    if Admins.query.count() == 0:
        admin_user = os.getenv('ADMIN_USERNAME')
        admin_pass = os.getenv('ADMIN_PASSWORD')
        if admin_user and admin_pass:
            db.session.add(Admins(username=admin_user, password_hash=generate_password_hash(admin_pass)))
            print(f"Created initial admin '{admin_user}' from environment.")
        else:
            print("No admin exists and ADMIN_USERNAME/ADMIN_PASSWORD are not set. "
                  "Skipping admin creation — set them in .env and re-run to create one.")

    db.session.commit()
    print("Database updated!")

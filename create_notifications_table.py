from app import app, db

with app.app_context():
    print("Creating tables (Notifications)...")
    db.create_all()
    print("Done.")

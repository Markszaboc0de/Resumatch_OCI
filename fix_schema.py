from app import app, db
from sqlalchemy import text

def fix_schema():
    with app.app_context():
        # Get engine from configured app
        engine = db.engine
        print(f"Connecting to {engine.url}...")
        
        with engine.connect() as conn:
            print("Checking/Adding 'filename' column...")
            try:
                conn.execute(text("ALTER TABLE cvs ADD COLUMN filename VARCHAR(255);"))
                print("Added 'filename' column.")
            except Exception as e:
                print(f"Column 'filename' might already exist or error: {e}")
                
            print("Checking/Adding 'file_path' column...")
            try:
                conn.execute(text("ALTER TABLE cvs ADD COLUMN file_path VARCHAR(512);"))
                print("Added 'file_path' column.")
            except Exception as e:
                print(f"Column 'file_path' might already exist or error: {e}")
            
            print("Checking/Adding 'is_main' column...")
            try:
                conn.execute(text("ALTER TABLE cvs ADD COLUMN is_main BOOLEAN DEFAULT FALSE;"))
                print("Added 'is_main' column.")
            except Exception as e:
                print(f"Column 'is_main' might already exist or error: {e}")

            conn.commit()
    print("Schema update complete.")

if __name__ == "__main__":
    fix_schema()

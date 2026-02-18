import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Use fallback if env var is missing (dev default)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://mac:csenger@localhost/resumatch')

def fix_schema():
    print(f"Connecting to {DATABASE_URL}...")
    engine = create_engine(DATABASE_URL)
    
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
            
        conn.commit()
    print("Schema update complete.")

if __name__ == "__main__":
    fix_schema()

import csv
import os
import pandas as pd
from dotenv import load_dotenv
from app import app, db, Job_Descriptions, clean_text, refresh_all_matches

# Load environment variables
load_dotenv()

CSV_FILE = 'jobs.csv'
BATCH_SIZE = 100

def populate_jobs():
    """
    Reads jobs.csv and populates the Job_Descriptions table.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print(f"Reading {CSV_FILE}...")
    
    # Try reading with pandas for robust CSV handling
    try:
        df = pd.read_csv(CSV_FILE, sep=';', on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        return

    # Normalize column names if needed (optional, assuming standard names)
    # Expected columns based on user request: company, title, city, country, raw_text, url
    
    new_objects = []
    total_inserted = 0
    
    with app.app_context():
        print("Connected to database.")
        
        for index, row in df.iterrows():
            # Extract fields with safe defaults - Mapped to actual CSV headers
            company = row.get('Company', None)
            title = row.get('Job Title', 'Unknown Title')
            city = row.get('City', None)
            country = row.get('Country', None)
            raw_text = row.get('Job Description', '')
            url = row.get('URL', None)
            
            # Simple validation
            if not raw_text:
                print(f"Skipping row {index}: Missing raw_text")
                continue

            # NLP Tokenization (using clean_text from app.py)
            parsed_tokens = clean_text(raw_text)

            # Create Job_Descriptions object
            job = Job_Descriptions(
                company=company,
                title=title,
                city=city,
                country=country,
                raw_text=raw_text,
                url=url,
                parsed_tokens=parsed_tokens,
                is_intern=False,  # Default
                active_status=True # Default
            )
            
            new_objects.append(job)

            # Batch Insert
            if len(new_objects) >= BATCH_SIZE:
                try:
                    db.session.bulk_save_objects(new_objects)
                    db.session.commit()
                    total_inserted += len(new_objects)
                    print(f"Inserted batch of {len(new_objects)} jobs. Total: {total_inserted}")
                    new_objects = [] # Reset batch
                except Exception as e:
                    db.session.rollback()
                    print(f"Error inserting batch: {e}")
                    new_objects = [] # Reset to avoid infinite loop on bad batch

        # Insert remaining
        if new_objects:
            try:
                db.session.bulk_save_objects(new_objects)
                db.session.commit()
                total_inserted += len(new_objects)
                print(f"Inserted final batch. Total: {total_inserted}")
            except Exception as e:
                db.session.rollback()
                print(f"Error inserting final batch: {e}")

    print("Population complete.")
    
    # Trigger Score Refresh
    print("\nTriggering score refresh...")
    refresh_all_matches()

if __name__ == "__main__":
    populate_jobs()

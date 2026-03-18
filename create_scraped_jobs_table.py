import os
import psycopg2
from urllib.parse import urlparse

# Connect to the database using the same DATABASE_URL as the main app
# Or fallback to the default connection string if not found
DATABASE_URL = os.getenv(
    'DATABASE_URL', 
    "postgresql://app_user:Mindenszarhoz@localhost:5432/job_match_db"
)

def create_scraped_jobs_table():
    print(f"Connecting to database: {DATABASE_URL}")
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Define the SQL to drop (if exists) and create the table
        sql_commands = """
        DROP TABLE IF EXISTS scraped_jobs;
        
        CREATE TABLE scraped_jobs (
            jd_id SERIAL PRIMARY KEY,
            company VARCHAR(255),
            title VARCHAR(255),
            city VARCHAR(255),
            country VARCHAR(255),
            raw_text TEXT,
            url TEXT
        );
        """
        
        # Execute the commands
        cur.execute(sql_commands)
        
        # Commit the changes and close
        conn.commit()
        cur.close()
        conn.close()
        
        print("Successfully created the 'scraped_jobs' table on the server!")
    except Exception as e:
        print(f"Error creating table: {e}")

if __name__ == '__main__':
    create_scraped_jobs_table()

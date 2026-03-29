"""
Migration script: Create the 'surveys' table.
Run once to add the surveys table to the database.
"""
import os
import sqlalchemy
from sqlalchemy import text

DATABASE_URL = os.getenv('DATABASE_URL', "postgresql://app_user:Mindenszarhoz@localhost:5432/job_match_db")

engine = sqlalchemy.create_engine(DATABASE_URL)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS surveys (
            survey_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
            survey_name VARCHAR(255) NOT NULL,
            survey_description VARCHAR(500) NOT NULL,
            survey_url TEXT NOT NULL,
            estimated_minutes INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
    conn.commit()
    print("✅ surveys table created successfully.")

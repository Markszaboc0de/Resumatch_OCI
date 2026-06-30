import os
import sys
# Add parent directory to path to find 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.extensions import db
from sqlalchemy import text

app = create_app()

with app.app_context():
    try:
        db.session.execute(text(
            'ALTER TABLE users ADD COLUMN IF NOT EXISTS token_balance INTEGER NOT NULL DEFAULT 0'
        ))
        db.session.execute(text(
            'ALTER TABLE users ADD COLUMN IF NOT EXISTS surveys_filled_count INTEGER NOT NULL DEFAULT 0'
        ))
        db.session.commit()
        print("Added token_balance and surveys_filled_count to users.")

        db.session.execute(text('''
            CREATE TABLE IF NOT EXISTS survey_completions (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(user_id),
                survey_id   INTEGER NOT NULL REFERENCES surveys(survey_id),
                started_at  TIMESTAMP NOT NULL,
                claimed_at  TIMESTAMP,
                tokens_awarded INTEGER,
                UNIQUE (user_id, survey_id)
            )
        '''))
        db.session.commit()
        print("Created survey_completions table.")

        db.session.execute(text('''
            CREATE TABLE IF NOT EXISTS token_transactions (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(user_id),
                amount     INTEGER NOT NULL,
                reason     VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        '''))
        db.session.commit()
        print("Created token_transactions table.")

        print("Migration complete.")
    except Exception as e:
        db.session.rollback()
        print(f"Error: {e}")

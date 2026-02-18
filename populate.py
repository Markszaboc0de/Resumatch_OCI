import csv
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from app import app, db, Job_Descriptions, CVs, Precalc_Scores, clean_text, nlp_model
from sentence_transformers import util

# Load environment variables
load_dotenv()

CSV_FILE = 'jobs_data.csv'
BATCH_SIZE = 100

def populate_jobs():
    """
    Reads jobs_data.csv, clears old data efficiently, and handles batch import + scoring.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print(f"Reading {CSV_FILE}...")
    
    try:
        df = pd.read_csv(CSV_FILE, sep=';', on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        return
        
    new_jobs_batch = []
    total_inserted = 0
    
    with app.app_context():
        print("Connected to database.")

        # 1. Fast Clear with TRUNCATE
        try:
            print("Clearing tables with TRUNCATE...")
            db.session.execute(text('TRUNCATE TABLE precalc_scores, job_descriptions RESTART IDENTITY CASCADE'))
            db.session.commit()
            print("Tables cleared.")
        except Exception as e:
            db.session.rollback()
            print(f"Error clearing tables: {e}")
            return
            
        # 2. Pre-fetch CVs and encode them ONCE
        print("Fetching and encoding CVs for matching...")
        cvs = CVs.query.all()
        cv_data = []
        if cvs:
            cv_texts = [clean_text(cv.raw_text) for cv in cvs]
            # Convert to list for efficient reuse
            cv_embeddings = nlp_model.encode(cv_texts, convert_to_tensor=True)
            cv_ids = [cv.cv_id for cv in cvs]
            print(f"Encoded {len(cvs)} CVs.")
        else:
            print("No CVs found. Skipping match calculation.")

        # 3. Stream Processing
        for index, row in df.iterrows():
            # Extract fields...
            company = row.get('Company', None)
            title = row.get('Job Title', 'Unknown Title')
            city = row.get('City', None)
            country = row.get('Country', None)
            raw_text = row.get('Job Description')
            url = row.get('URL', None)
            
            if pd.isna(raw_text) or str(raw_text).strip() == '':
                continue
            
            # Sanitize
            raw_text = str(raw_text).replace('\x00', '')
            if company: company = str(company).replace('\x00', '')
            if title: title = str(title).replace('\x00', '')
            if city: city = str(city).replace('\x00', '')
            if country: country = str(country).replace('\x00', '')

            # Calculate Tokens
            parsed_tokens = clean_text(raw_text)

            job = Job_Descriptions(
                company=company,
                title=title,
                city=city,
                country=country,
                raw_text=raw_text,
                url=url,
                parsed_tokens=parsed_tokens,
                is_intern=False,
                active_status=True
            )
            
            new_jobs_batch.append(job)

            # Process Batch
            if len(new_jobs_batch) >= BATCH_SIZE:
                try:
                    # A. Insert Jobs and generate IDs
                    db.session.add_all(new_jobs_batch)
                    db.session.flush() # Flushes to DB to get IDs, doesn't commit yet
                    
                    # B. Calculate Scores for this batch if CVs exist
                    if cvs and new_jobs_batch:
                        # Encode current batch of jobs
                        batch_job_texts = [j.parsed_tokens for j in new_jobs_batch] # Use parsed_tokens as it's already cleaned
                        batch_job_embeddings = nlp_model.encode(batch_job_texts, convert_to_tensor=True)
                        
                        # Calculate Sim Matrix [num_cvs, num_batch_jobs]
                        cosine_scores = util.cos_sim(cv_embeddings, batch_job_embeddings)
                        
                        new_scores = []
                        for i in range(len(cvs)):
                            for j in range(len(new_jobs_batch)):
                                score = float(cosine_scores[i][j])
                                new_scores.append(Precalc_Scores(
                                    cv_id=cv_ids[i],
                                    jd_id=new_jobs_batch[j].jd_id,
                                    similarity_score=score
                                ))
                        
                        # Bulk save scores
                        db.session.bulk_save_objects(new_scores)

                    # C. Commit everything
                    db.session.commit()
                    total_inserted += len(new_jobs_batch)
                    print(f"Processed batch of {len(new_jobs_batch)} jobs + scores. Total jobs: {total_inserted}")
                    new_jobs_batch = [] 

                except Exception as e:
                    db.session.rollback()
                    print(f"Error processing batch: {e}")
                    new_jobs_batch = []

        # Process Final Batch
        if new_jobs_batch:
            try:
                db.session.add_all(new_jobs_batch)
                db.session.flush()
                
                if cvs:
                    batch_job_texts = [j.parsed_tokens for j in new_jobs_batch]
                    batch_job_embeddings = nlp_model.encode(batch_job_texts, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(cv_embeddings, batch_job_embeddings)
                    
                    new_scores = []
                    for i in range(len(cvs)):
                        for j in range(len(new_jobs_batch)):
                            score = float(cosine_scores[i][j])
                            new_scores.append(Precalc_Scores(
                                cv_id=cv_ids[i],
                                jd_id=new_jobs_batch[j].jd_id,
                                similarity_score=score
                            ))
                    db.session.bulk_save_objects(new_scores)
                
                db.session.commit()
                total_inserted += len(new_jobs_batch)
                print(f"Processed final batch. Total: {total_inserted}")
            except Exception as e:
                db.session.rollback()
                print(f"Error final batch: {e}")

    print("Population and scoring complete.")

if __name__ == "__main__":
    populate_jobs()

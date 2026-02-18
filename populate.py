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

            # Filter: Only process 'intern' jobs
            if 'intern' not in str(title).lower():
                continue
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
                    
                    # B. Logic removed: No longer calculating scores per batch to avoid DB bloat.
                    
                    # C. Commit Job Batch
                    db.session.commit()
                    total_inserted += len(new_jobs_batch)
                    print(f"Processed batch of {len(new_jobs_batch)} jobs. Total jobs: {total_inserted}")
                    new_jobs_batch = [] 

                except Exception as e:
                    db.session.rollback()
                    print(f"Error processing batch: {e}")
                    new_jobs_batch = []

        # Process Final Batch
        if new_jobs_batch:
            try:
                db.session.add_all(new_jobs_batch)
                db.session.commit()
                total_inserted += len(new_jobs_batch)
                print(f"Processed final batch. Total: {total_inserted}")
            except Exception as e:
                db.session.rollback()
                print(f"Error final batch: {e}")

        # --- CACHING EMBEDDINGS & CALCULATING MATCHES (Top 10 Only) ---
        print("Calculating and Caching Job Embeddings...")
        try:
            import torch
            
            # Fetch all active jobs (we just inserted them)
            jobs = Job_Descriptions.query.filter_by(active_status=True).all()
            if jobs:
                print(f"Encoding {len(jobs)} jobs for cache...")
                job_texts = [clean_text(job.raw_text) for job in jobs]
                job_ids = [job.jd_id for job in jobs]
                
                # Encode Jobs
                job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)
                
                # Save to file
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
                torch.save({
                    'embeddings': job_embeddings,
                    'ids': job_ids
                }, output_path)
                print(f"Saved job embeddings to {output_path}")

                # --- MATCHING Logic (If CVs exist) ---
                if cvs:
                    print(f"Calculating Top 10 matches for {len(cvs)} CVs...")
                    
                    # Compute Similarity Matrix [N_CVs, M_Jobs]
                    # cv_embeddings was computed at the start
                    cosine_scores = util.cos_sim(cv_embeddings, job_embeddings)
                    
                    new_scores_objects = []
                    
                    for i, cv in enumerate(cvs):
                        # Get scores for this CV against all jobs
                        cv_scores = cosine_scores[i]
                        
                        # Pair with Job IDs and Score
                        cv_matches = []
                        for j, score in enumerate(cv_scores):
                            cv_matches.append({
                                'cv_id': cv_ids[i], # cv_ids list from start
                                'jd_id': job_ids[j],
                                'similarity_score': float(score)
                            })
                        
                        # Sort and Keep Global Top 10
                        top_10 = sorted(cv_matches, key=lambda x: x['similarity_score'], reverse=True)[:10]
                        
                        for match in top_10:
                            new_scores_objects.append(Precalc_Scores(**match))
                    
                    # Batch Insert Scores
                    try:
                        # Clear any existing scores (should be empty due to TRUNCATE but safe to do)
                        db.session.query(Precalc_Scores).delete()
                        db.session.bulk_save_objects(new_scores_objects)
                        db.session.commit()
                        print(f"Saved {len(new_scores_objects)} top matches to Precalc_Scores.")
                    except Exception as e:
                        db.session.rollback()
                        print(f"Error saving matches: {e}")

            else:
                print("No jobs found to cache.")
                
        except Exception as e:
            print(f"Error caching/matching: {e}")
            import traceback
            traceback.print_exc()

    print("Population and scoring complete.")

if __name__ == "__main__":
    populate_jobs()

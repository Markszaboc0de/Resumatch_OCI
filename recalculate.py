import os
import json
from app import app, db, CVs, Job_Descriptions, extract_text_from_pdf, extract_text_from_txt, clean_text, nlp_model, refresh_all_matches, extract_skills_from_text

def recalculate_cvs():
    with app.app_context():
        cvs = CVs.query.all()
        print(f"Found {len(cvs)} CVs to process.")
        
        for cv in cvs:
            if cv.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], cv.filename)
                if os.path.exists(file_path):
                    if cv.filename.lower().endswith('.pdf'):
                        text = extract_text_from_pdf(file_path)
                    elif cv.filename.lower().endswith('.txt'):
                        text = extract_text_from_txt(file_path)
                    else:
                        continue
                    cv.raw_text = text
                    print(f"Extracted text for CV ID {cv.cv_id} from {cv.filename}")
                else:
                    print(f"File missing for CV {cv.cv_id}: {file_path}")
            
            if cv.raw_text:
                cleaned = clean_text(cv.raw_text).replace('\x00', '')
                cv.extracted_skills = extract_skills_from_text(cleaned)
                vec = nlp_model.encode(cleaned, convert_to_tensor=False)
                cv.parsed_tokens = json.dumps(vec.tolist())
                print(f"Re-embedded CV ID {cv.cv_id}")
        
        db.session.commit()
        print("CV Embeddings and texts updated!")

def recalculate_jobs():
    with app.app_context():
        jobs = Job_Descriptions.query.all()
        print(f"Found {len(jobs)} Jobs to process.")
        for job in jobs:
            if job.raw_text:
                if not job.extracted_skills or not job.parsed_tokens or not job.parsed_tokens.startswith('['):
                    cleaned = clean_text(job.raw_text).replace('\x00', '')
                    job.extracted_skills = extract_skills_from_text(cleaned)
                    vec = nlp_model.encode(cleaned, convert_to_tensor=False)
                    job.parsed_tokens = json.dumps(vec.tolist())
                    print(f"Computed Vector & Skills for Job {job.jd_id}")
        db.session.commit()
        
        # Also remove the cached job embeddings file to force re-generation
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("Deleted cached job embeddings (will be auto-regenerated).")
        print("Job Embeddings text updated!")

if __name__ == "__main__":
    recalculate_cvs()
    recalculate_jobs()
    print("Recalculating all matches table...")
    refresh_all_matches()
    print("Done!")

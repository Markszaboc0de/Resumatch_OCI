import os
import json
from app import (
    app, db, CVs, Job_Descriptions,
    extract_text_from_pdf, extract_text_from_txt, clean_text,
    nlp_model, refresh_all_matches, extract_skills_from_text, logger,
)

def recalculate_cvs():
    with app.app_context():
        cvs = CVs.query.all()
        logger.info("Found %d CVs to process.", len(cvs))

        for cv in cvs:
            # Legacy path: older CVs may have a file on disk. The current upload flow
            # stores the extracted text directly in cv.raw_text and never writes a file,
            # so a missing file here is expected (not an error) — we just re-embed raw_text.
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
                    logger.debug("Re-extracted text for CV ID %s from %s", cv.cv_id, cv.filename)
                else:
                    logger.debug("No file on disk for CV %s (text held in DB): %s", cv.cv_id, file_path)

            if cv.raw_text:
                cleaned = clean_text(cv.raw_text).replace('\x00', '')
                cv.extracted_skills = extract_skills_from_text(cleaned)
                vec = nlp_model.encode(cleaned, convert_to_tensor=False)
                cv.parsed_tokens = json.dumps(vec.tolist())
                logger.debug("Re-embedded CV ID %s", cv.cv_id)

        db.session.commit()
        logger.info("CV embeddings and texts updated.")

def recalculate_jobs():
    with app.app_context():
        jobs = Job_Descriptions.query.all()
        logger.info("Found %d jobs to process.", len(jobs))
        for job in jobs:
            if job.raw_text:
                if not job.extracted_skills or not job.parsed_tokens or not job.parsed_tokens.startswith('['):
                    cleaned = clean_text(job.raw_text).replace('\x00', '')
                    job.extracted_skills = extract_skills_from_text(cleaned)
                    vec = nlp_model.encode(cleaned, convert_to_tensor=False)
                    job.parsed_tokens = json.dumps(vec.tolist())
                    logger.debug("Computed vector & skills for Job %s", job.jd_id)
        db.session.commit()

        # Also remove the cached job embeddings file to force re-generation
        cache_path = os.path.join(app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info("Deleted cached job embeddings (will be auto-regenerated).")
        logger.info("Job embeddings updated.")

if __name__ == "__main__":
    recalculate_cvs()
    recalculate_jobs()
    logger.info("Recalculating all matches table...")
    refresh_all_matches()
    logger.info("Done.")

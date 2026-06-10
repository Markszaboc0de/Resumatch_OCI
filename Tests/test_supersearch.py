from app import app, db, check_and_deduct_supersearch, clean_text, nlp_model, load_cached_embeddings, Job_Descriptions
from sentence_transformers import util

def test_super():
    with app.app_context():
        ideal_desc = "Test description"
        cleaned_desc = clean_text(ideal_desc)
        ideal_embedding = nlp_model.encode(cleaned_desc, convert_to_tensor=True)

        job_embeddings, job_ids = load_cached_embeddings()
        active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
        
        if job_embeddings is None:
            job_texts = [clean_text(job.raw_text) for job in active_jobs]
            job_ids = [job.jd_id for job in active_jobs]
            job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)
        
        scores = util.cos_sim(ideal_embedding, job_embeddings)[0]
        
        active_job_ids = {job.jd_id: job for job in active_jobs}
        
        all_scores = []
        for idx, score in enumerate(scores):
            if job_ids[idx] in active_job_ids:
                all_scores.append((float(score), job_ids[idx]))

        top_matches = sorted(all_scores, key=lambda x: x[0], reverse=True)[:3]
        print("Success! Matches:", top_matches)

if __name__ == "__main__":
    test_super()

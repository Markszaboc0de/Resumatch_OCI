"""AI / matching service.

Loads the NLP + NER models once at import (so gunicorn ``--preload`` shares them
across workers) and exposes the scoring functions plus the Celery tasks that run
the heavy background work.

Compared with the original app.py, the only behavioural change is *how* the
background work is dispatched: ``threading.Thread`` is replaced by Celery tasks
(``run_jit_task`` / ``run_upload_jit_full_task``). The scoring logic — hybrid
semantic+NER score and the Top-3 global / Hungarian / abroad buckets — is
preserved verbatim.
"""
import os
import json
import time
import logging
import tempfile
from datetime import datetime

from sentence_transformers import SentenceTransformer, util
from gliner import GLiNER
from flask import current_app

from app.extensions import db, celery
from app.models import Users, CVs, Job_Descriptions, Precalc_Scores
from app.utils.helpers import (
    clean_text,
    mark_calculating,
    clear_calculating,
    is_user_calculating,
)

logger = logging.getLogger('resumatch')

# Load NLP Model (Global) - Loaded once at startup
logger.info("Loading NLP & NER Models...")
nlp_model = SentenceTransformer("BAAI/bge-m3")
ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
logger.info("Models Loaded.")


def extract_skills_from_text(text):
    if not text:
        return "[]"
    try:
        labels = ["Skill", "Tool", "Technology", "Framework", "Software"]
        # Max length to avoid massive memory usage or slow inference
        ents = ner_model.predict_entities(text[:3000].lower(), set(labels))
        # Deduplicate
        skills = {ent['text'].strip() for ent in ents if len(ent['text'].strip()) >= 2}
        return json.dumps(list(skills))
    except Exception as e:
        logger.error(f"NER Error: {e}")
        return "[]"


def extract_match_reasons(cv_skills_json, jd_skills_json):
    """Extract the top overlapping skills between a CV and a Job Description."""
    if not cv_skills_json or not jd_skills_json:
        return ["No exact skill matches found"]

    try:
        cv_skills = set(json.loads(cv_skills_json))
        jd_skills = set(json.loads(jd_skills_json))

        overlap = cv_skills.intersection(jd_skills)
        if not overlap:
            return ["No exact skill matches found"]

        return list(overlap)[:3]
    except Exception:
        return ["No exact skill matches found"]


def load_cached_embeddings():
    """Helper to load pre-calculated job embeddings."""
    try:
        import torch
        cache_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        if os.path.exists(cache_path):
            logger.debug(f"Loading cached embeddings from {cache_path}...")
            cache = torch.load(cache_path)
            return cache.get('embeddings'), cache.get('ids')
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
    return None, None


def save_embeddings_cache(job_embeddings, job_ids):
    """Atomically persist the job-embeddings cache.

    Writes to a temp file in the same directory then os.replace() (atomic on
    POSIX), so concurrent gunicorn workers/threads can never read a half-written
    file.
    """
    import torch
    cache_dir = current_app.config['UPLOAD_FOLDER']
    cache_path = os.path.join(cache_dir, 'job_embeddings.pt')
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, prefix='.job_embeddings_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save({'embeddings': job_embeddings, 'ids': job_ids}, f)
        os.replace(tmp_path, cache_path)  # atomic swap
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def calculate_matches_background(cv_id, cv_embedding_list):
    from app import app  # lazy import avoids a circular import at module load
    with app.app_context():
        import torch
        start_time = time.time()
        logger.debug(f"Starting background matching for CV ID: {cv_id}")

        # 1. Check for Cached Embeddings
        job_embeddings, job_ids = load_cached_embeddings()

        if job_embeddings is None:
            logger.info("Cache miss! Rebuilding cache from DB vectors...")
            active_jobs = Job_Descriptions.query.filter_by(active_status=True).all()
            if not active_jobs:
                logger.warning("No active jobs found for matching.")
                return

            job_embeddings_list = []
            job_ids = []
            for job in active_jobs:
                if job.parsed_tokens and job.parsed_tokens.startswith('['):
                    try:
                        job_embeddings_list.append(json.loads(job.parsed_tokens))
                        job_ids.append(job.jd_id)
                    except Exception:
                        logger.debug("Skipping job %s: unparseable parsed_tokens", job.jd_id, exc_info=True)

            if not job_embeddings_list:
                logger.warning("No valid encoded jobs found in DB! Make sure you run recalculate.py first.")
                return

            job_embeddings = torch.tensor(job_embeddings_list)

            try:
                save_embeddings_cache(job_embeddings, job_ids)
                logger.info(f"Successfully rebuilt job_embeddings.pt cache with {len(job_ids)} jobs.")
            except Exception as e:
                logger.warning(f"Failed to save temporary cache: {e}")
        else:
            logger.debug(f"Loaded cached embeddings for {len(job_ids)} jobs (FAST).")

        # 2. Encode CV
        logger.debug("Using Pre-encoded User CV...")
        cv_embedding = torch.tensor(cv_embedding_list).unsqueeze(0)  # [1, vector_dim]

        # 3. Calculate Cosine Similarity -> [1, N]
        scores = util.cos_sim(cv_embedding, job_embeddings)[0]

        # Fetch skills for Hybrid Scoring
        cv = CVs.query.get(cv_id)
        cv_skills = set(json.loads(cv.extracted_skills or "[]")) if cv else set()
        active_jobs_for_skills = Job_Descriptions.query.filter(Job_Descriptions.jd_id.in_(job_ids)).all()
        job_skills_map = {job.jd_id: set(json.loads(job.extracted_skills or "[]")) for job in active_jobs_for_skills}
        job_country_map = {job.jd_id: str(job.country).lower() if job.country else '' for job in active_jobs_for_skills}

        # 4. Prepare Scores List
        all_scores = []
        for idx, score in enumerate(scores):
            semantic_score = float(score)
            jd_id = job_ids[idx]

            job_skills = job_skills_map.get(jd_id, set())
            if len(job_skills) == 0:
                final_score = semantic_score
            else:
                overlap = cv_skills.intersection(job_skills)
                ner_score = len(overlap) / len(job_skills)
                final_score = (semantic_score * 0.7) + (ner_score * 0.3)

            all_scores.append({
                'cv_id': cv_id,
                'jd_id': jd_id,
                'similarity_score': final_score
            })

        # 5. SORT and KEEP TOP 3 GLOBAL, TOP 3 HUNGARIAN, and TOP 3 ABROAD
        # This keeps the DB tiny while ensuring the location filter works.
        sorted_scores = sorted(all_scores, key=lambda x: x['similarity_score'], reverse=True)
        top_matches = []
        global_added = 0
        hungarian_added = 0
        abroad_added = 0

        for match in sorted_scores:
            jd_id = match['jd_id']
            country = job_country_map.get(jd_id, '')
            is_hungary = 'hungary' in country or 'magyarország' in country
            is_abroad = not is_hungary

            if global_added < 3:
                top_matches.append(match)
                global_added += 1
                if is_hungary:
                    hungarian_added += 1
                if is_abroad:
                    abroad_added += 1
            elif is_hungary and hungarian_added < 3:
                top_matches.append(match)
                hungarian_added += 1
            elif is_abroad and abroad_added < 3:
                top_matches.append(match)
                abroad_added += 1

            if global_added >= 3 and hungarian_added >= 3 and abroad_added >= 3:
                break

        # 6. Insert into Precalc_Scores
        new_scores_objects = [Precalc_Scores(**match) for match in top_matches]

        try:
            # Clear existing scores for this CV to avoid duplicates if re-running
            Precalc_Scores.query.filter_by(cv_id=cv_id).delete()

            db.session.bulk_save_objects(new_scores_objects)
            db.session.commit()
            logger.info(f"Successfully calculated and saved Top {len(new_scores_objects)} matches for CV ID: {cv_id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving scores for CV ID {cv_id}: {e}")


def refresh_all_matches():
    """Recalculate match scores for ALL CVs against ALL active Job Descriptions.

    Used after bulk data updates (e.g. recalculate.py).
    """
    from app import app  # lazy import avoids a circular import at module load
    with app.app_context():
        import torch
        logger.info("Starting full score refresh...")

        # 1. Fetch Data
        cvs = CVs.query.all()
        jobs = Job_Descriptions.query.filter_by(active_status=True).all()

        if not cvs or not jobs:
            logger.warning("Insufficient data (CVs or Jobs) to calculate matches.")
            return

        logger.info(f"Matches to compute: {len(cvs)} CVs x {len(jobs)} Jobs")

        # 2. Prepare Embeddings — encode CVs
        cv_embeddings_list = []
        cv_ids = []

        for cv in cvs:
            if cv.parsed_tokens and cv.parsed_tokens.startswith('['):
                try:
                    vec = json.loads(cv.parsed_tokens)
                    cv_embeddings_list.append(vec)
                    cv_ids.append(cv.cv_id)
                except Exception:
                    logger.debug("Skipping CV %s: unparseable parsed_tokens", cv.cv_id, exc_info=True)
            elif cv.raw_text:  # Fallback to encoding old texts if necessary
                vec = nlp_model.encode(clean_text(cv.raw_text), convert_to_tensor=False)
                cv_embeddings_list.append(vec.tolist())
                cv_ids.append(cv.cv_id)

        if not cv_embeddings_list:
            logger.warning("No valid CV vectors found.")
            return

        cv_embeddings = torch.tensor(cv_embeddings_list)

        # Load Job Embeddings from DB
        job_embeddings_list = []
        job_ids = []
        for job in jobs:
            if job.parsed_tokens and job.parsed_tokens.startswith('['):
                try:
                    job_embeddings_list.append(json.loads(job.parsed_tokens))
                    job_ids.append(job.jd_id)
                except Exception:
                    logger.debug("Skipping job %s: unparseable parsed_tokens", job.jd_id, exc_info=True)

        if not job_embeddings_list:
            logger.warning("No valid encoded jobs found! Run recalculate.py first.")
            return

        job_embeddings = torch.tensor(job_embeddings_list)

        try:
            save_embeddings_cache(job_embeddings, job_ids)
            logger.info("Saved updated job embeddings to cache.")
        except Exception as e:
            logger.warning(f"Failed to save updated job cache: {e}")

        # 3. Compute Similarity Matrix [N_CVs, M_Jobs]
        if len(cv_embeddings) > 0 and len(job_embeddings) > 0:
            all_scores_matrix = util.cos_sim(cv_embeddings, job_embeddings)

            job_skills_map = {job.jd_id: set(json.loads(job.extracted_skills or "[]")) for job in jobs}
            job_country_map = {job.jd_id: str(job.country).lower() if job.country else '' for job in jobs}

            new_scores_objects = []

            for i, cv in enumerate(cvs):
                # cv_ids/cv_embeddings may skip CVs without vectors; guard index
                if i >= len(all_scores_matrix):
                    break
                cv_scores = all_scores_matrix[i]
                cv_skills = set(json.loads(cv.extracted_skills or "[]"))

                cv_matches = []
                for j, score in enumerate(cv_scores):
                    semantic_score = float(score)
                    jd_id = job_ids[j]

                    job_skills = job_skills_map.get(jd_id, set())
                    if len(job_skills) == 0:
                        final_score = semantic_score
                    else:
                        overlap = cv_skills.intersection(job_skills)
                        ner_score = len(overlap) / len(job_skills)
                        final_score = (semantic_score * 0.7) + (ner_score * 0.3)

                    cv_matches.append({
                        'cv_id': cv.cv_id,
                        'jd_id': jd_id,
                        'similarity_score': final_score
                    })

                # Sort and Keep Top 3 GLOBAL, TOP 3 HUNGARIAN, and TOP 3 ABROAD
                sorted_scores = sorted(cv_matches, key=lambda x: x['similarity_score'], reverse=True)
                global_added = 0
                hungarian_added = 0
                abroad_added = 0

                for match in sorted_scores:
                    jd_id = match['jd_id']
                    country = job_country_map.get(jd_id, '')
                    is_hungary = 'hungary' in country or 'magyarország' in country
                    is_abroad = not is_hungary

                    if global_added < 3:
                        new_scores_objects.append(Precalc_Scores(**match))
                        global_added += 1
                        if is_hungary:
                            hungarian_added += 1
                        if is_abroad:
                            abroad_added += 1
                    elif is_hungary and hungarian_added < 3:
                        new_scores_objects.append(Precalc_Scores(**match))
                        hungarian_added += 1
                    elif is_abroad and abroad_added < 3:
                        new_scores_objects.append(Precalc_Scores(**match))
                        abroad_added += 1

                    if global_added >= 3 and hungarian_added >= 3 and abroad_added >= 3:
                        break

            # Batch Insert
            try:
                db.session.query(Precalc_Scores).delete()
                db.session.bulk_save_objects(new_scores_objects)
                db.session.commit()
                logger.info(f"Full Refresh Logic Complete. Saved {len(new_scores_objects)} top matches.")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error saving refresh matches: {e}")


# --- Celery tasks (replace the original threading.Thread background work) ---

@celery.task
def run_jit_task(user_id):
    """Re-evaluate all of a user's CVs against the current job set."""
    try:
        user_cvs = CVs.query.filter_by(user_id=user_id).all()
        for cv in user_cvs:
            if cv.parsed_tokens:
                vec = json.loads(cv.parsed_tokens)
                calculate_matches_background(cv.cv_id, vec)
    finally:
        clear_calculating(user_id)


@celery.task
def run_upload_jit_full_task(user_id, cv_id, clean_txt):
    """Full pipeline for a freshly uploaded CV: NER + embedding + matching."""
    try:
        # A. Run AI Models (Slow)
        cv_skills_json = extract_skills_from_text(clean_txt)
        cv_embedding = nlp_model.encode(clean_txt, convert_to_tensor=False)
        vector_json = json.dumps(cv_embedding.tolist())

        # B. Update DB with AI results
        cv_to_update = CVs.query.get(cv_id)
        if cv_to_update:
            cv_to_update.extracted_skills = cv_skills_json
            cv_to_update.parsed_tokens = vector_json
            db.session.commit()

        # C. Run Matching
        calculate_matches_background(cv_id, cv_embedding.tolist())
    except Exception:
        logger.exception("Background AI analysis failed for CV %s", cv_id)
    finally:
        clear_calculating(user_id)


def ensure_jit_matches(user):
    """If the job cache changed since the user was last active (or is missing),
    dispatch a background recalculation via Celery.
    """
    try:
        if is_user_calculating(user.user_id):
            return

        cache_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'job_embeddings.pt')
        needs_recalc = False

        if not os.path.exists(cache_path):
            needs_recalc = True
        else:
            cache_mtime = os.path.getmtime(cache_path)
            cache_time = datetime.utcfromtimestamp(cache_mtime)
            if not user.last_active_date or user.last_active_date < cache_time:
                needs_recalc = True

        if needs_recalc:
            logger.info(f"JIT Matching trigger: Re-evaluating CVs for User {user.user_id}")
            mark_calculating(user.user_id)
            run_jit_task.delay(user.user_id)

            # Prevent infinite recalculation loop by immediately updating last_active_date
            user.last_active_date = datetime.utcnow()
            db.session.commit()
    except Exception as e:
        logger.error(f"JIT matching evaluation failed: {e}")
        clear_calculating(user.user_id)

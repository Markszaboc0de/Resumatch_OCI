from app import app, db, CVs, Job_Descriptions, Precalc_Scores, Users

def verify():
    with app.app_context():
        print(f"--- Database Verification ---")
        
        # Check Job Descriptions
        job_count = Job_Descriptions.query.count()
        print(f"Total Jobs: {job_count}")
        if job_count > 0:
            job = Job_Descriptions.query.first()
            print(f"Sample Job (ID {job.jd_id}): {job.title} at {job.company}")
            print(f"Sample Job Text Length: {len(job.raw_text)} chars")
        else:
            print("WARNING: Job_Descriptions table is EMPTY.")

        # Check CVs
        cv_count = CVs.query.count()
        print(f"Total CVs: {cv_count}")
        if cv_count > 0:
            cv = CVs.query.order_by(CVs.upload_date.desc()).first()
            print(f"Latest CV (ID {cv.cv_id}) User ID: {cv.user_id}")
            print(f"Latest CV Text Length: {len(cv.raw_text)} chars")
        else:
            print("WARNING: CVs table is EMPTY.")

        # Check Scores
        score_count = Precalc_Scores.query.count()
        print(f"Total Scores: {score_count}")
        if score_count > 0:
            score = Precalc_Scores.query.first()
            print(f"Sample Score: CV {score.cv_id} <-> Job {score.jd_id} = {score.similarity_score}")
        else:
            print("WARNING: Precalc_Scores table is EMPTY.")
            if cv_count > 0 and job_count > 0:
                print("ACTION REQUIRED: Data exists but scores are missing. Trigger refresh.")

        # Check Users
        user_count = Users.query.count()
        print(f"Total Users: {user_count}")

if __name__ == "__main__":
    verify()

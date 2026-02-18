from app import app, Job
from sqlalchemy import func

def verify_urls():
    with app.app_context():
        # Get count of jobs with URL
        total_jobs = Job.query.count()
        jobs_with_url = Job.query.filter(Job.url.isnot(None), Job.url != '').count()
        
        print(f"Total Jobs: {total_jobs}")
        print(f"Jobs with URL: {jobs_with_url}")
        
        # Print a sample
        sample_job = Job.query.filter(Job.url.isnot(None)).first()
        if sample_job:
            print(f"Sample Job ID: {sample_job.id}")
            print(f"Sample Job Title: {sample_job.title}")
            print(f"Sample Job URL: {sample_job.url}")
        else:
            print("No jobs with URL found.")

if __name__ == "__main__":
    verify_urls()

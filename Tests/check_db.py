from app import app, db, Job_Descriptions
with app.app_context():
    all_jobs = Job_Descriptions.query.all()
    print(f"Total jobs: {len(all_jobs)}")
    manual_jobs = Job_Descriptions.query.filter_by(is_direct_upload=True).count()
    scraper_jobs = Job_Descriptions.query.filter_by(is_direct_upload=False).count()
    print(f"Manual (is_direct_upload=True): {manual_jobs}")
    print(f"Scraper (is_direct_upload=False): {scraper_jobs}")
    null_jobs = Job_Descriptions.query.filter(Job_Descriptions.is_direct_upload == None).count()
    print(f"Null (is_direct_upload=NULL): {null_jobs}")

from app import app, populate_jobs, populate_resumes, db, Job, Resume

def reload_all_data():
    with app.app_context():
        print("Starting data reload...")
        populate_jobs(clear=True)
        populate_resumes(clear=True) # Optional but good for consistency
        print("Data reload complete.")

if __name__ == "__main__":
    reload_all_data()

from app import app, db, Job_Descriptions
from sqlalchemy import text
from recalculate import recalculate_jobs, refresh_all_matches

def sync_scraped_jobs():
    with app.app_context():
        print("--- Step A: Mark Phase ---")
        # Set sync_active = False for all external jobs
        db.session.execute(text("UPDATE job_descriptions SET sync_active = FALSE WHERE is_direct_upload = FALSE"))
        
        print("--- Step B: Process/Upsert Phase ---")
        # Read from scraped_jobs table
        try:
            sql_query = """
                SELECT jd_id, company, title, city, country, raw_text, url 
                FROM scraped_jobs 
                WHERE title ~* '\\b(intern|internship|entry-level|entry level|trainee|junior|graduate)\\b'
                   OR raw_text ~* '\\b(intern|internship|entry-level|entry level|trainee|junior|graduate)\\b'
            """
            result = db.session.execute(text(sql_query))
            rows = result.fetchall()
        except Exception as e:
            print("Error reading from scraped_jobs. Does the table exist? Run create_scraped_jobs_table.py first.")
            print(f"Details: {e}")
            return
            
        updated_count = 0
        inserted_count = 0
        
        for row in rows:
            _, company, title, city, country, raw_text, url = row
            
            # Check if this precise URL already exists
            existing_job = None
            if url:
                existing_job = Job_Descriptions.query.filter_by(url=url).first()
                
            if existing_job:
                # Update existing record
                existing_job.sync_active = True
                existing_job.company = company
                existing_job.title = title
                existing_job.city = city
                existing_job.country = country
                existing_job.raw_text = raw_text
                updated_count += 1
            else:
                # Insert new external job
                new_job = Job_Descriptions(
                    title=title,
                    company=company,
                    city=city,
                    country=country,
                    raw_text=raw_text,
                    url=url,
                    is_intern=False,
                    active_status=True,
                    is_native=False,
                    is_direct_upload=False,
                    sync_active=True
                )
                db.session.add(new_job)
                inserted_count += 1
                
        # Flush or commit upserts
        db.session.commit()
        print(f"Upsert phase complete. Updated: {updated_count}, Inserted: {inserted_count}")
        
        print("--- Step C: Cleanup/Sweep Phase ---")
        # Delete external jobs that weren't in the incoming payload
        # Find them first so we can report how many are deleted
        deleted_count = db.session.execute(text("DELETE FROM job_descriptions WHERE sync_active = FALSE AND is_direct_upload = FALSE")).rowcount
        
        # Clear the staging table
        db.session.execute(text("DELETE FROM scraped_jobs"))
        db.session.commit()
        
        print(f"Sweep complete. Deleted {deleted_count} stale external jobs.")
        print("Cleaned up scraped_jobs staging table.")
        
        # Recalculate if there were changes
        if updated_count > 0 or inserted_count > 0 or deleted_count > 0:
            print("Triggering recalculation and match refresh...")
            recalculate_jobs()
            refresh_all_matches()
        else:
            print("No changes detected. Skipping recalculation to avoid unnecessary CPU load.")

        print("Synchronization completed successfully!")

if __name__ == "__main__":
    sync_scraped_jobs()

from app import app, db, Job_Descriptions
from sqlalchemy import text
from recalculate import recalculate_jobs

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
                WHERE (
                       title ~* '\\y(intern|internship|entry-level|entry level|trainee|junior|graduate|gyakornok|kezdő|pályakezdő|friss diplomás)\\y'
                    OR raw_text ~* '\\y(intern|internship|entry-level|entry level|trainee|junior|graduate|gyakornok|kezdő|pályakezdő|friss diplomás)\\y'
                )
                AND (
                       country ~* '\\y(Austria|Belgium|Bulgaria|Croatia|Cyprus|Czechia|Czech Republic|Denmark|Estonia|Finland|France|Germany|Greece|Hungary|Ireland|Italy|Latvia|Lithuania|Luxembourg|Malta|Netherlands|Poland|Portugal|Romania|Slovakia|Slovenia|Spain|Sweden|United Kingdom|Switzerland|Norway|Europe)\\y'
                    OR raw_text ~* '\\y(Austria|Belgium|Bulgaria|Croatia|Cyprus|Czechia|Czech Republic|Denmark|Estonia|Finland|France|Germany|Greece|Hungary|Ireland|Italy|Latvia|Lithuania|Luxembourg|Malta|Netherlands|Poland|Portugal|Romania|Slovakia|Slovenia|Spain|Sweden|United Kingdom|Switzerland|Norway|Europe)\\y'
                )
                AND title !~* '\\y(director|senior|sr.|expert|president|associate|oktató|lead|clinical|head |VP)\\y'
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
                if existing_job.raw_text != raw_text:
                    existing_job.raw_text = raw_text
                    existing_job.parsed_tokens = None
                    existing_job.extracted_skills = None
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
        # 1. First cascade deletions for any dependent matching scores or notifications
        # This prevents the psycopg2 ForeignKeyViolation error.
        stale_jobs_query = """
            SELECT jd_id FROM job_descriptions 
            WHERE sync_active = FALSE AND is_direct_upload = FALSE
        """
        db.session.execute(text(f"DELETE FROM precalc_scores WHERE jd_id IN ({stale_jobs_query})"))
        db.session.execute(text(f"DELETE FROM notifications WHERE job_id IN ({stale_jobs_query})"))

        # 2. Safely delete the stale external jobs now that constraints are clear
        # Find them first so we can report how many are deleted
        deleted_count = db.session.execute(text("DELETE FROM job_descriptions WHERE sync_active = FALSE AND is_direct_upload = FALSE")).rowcount
        
        # Clear the staging table
        db.session.execute(text("DELETE FROM scraped_jobs"))
        db.session.commit()
        
        print(f"Sweep complete. Deleted {deleted_count} stale external jobs.")
        print("Cleaned up scraped_jobs staging table.")
        
        # Recalculate if there were changes
        if updated_count > 0 or inserted_count > 0 or deleted_count > 0:
            print("Triggering cache rebuild for new jobs...")
            recalculate_jobs()
            # Match calculation is now handled JIT during user login
        else:
            print("No changes detected. Skipping recalculation to avoid unnecessary CPU load.")

        print("Synchronization completed successfully!")

if __name__ == "__main__":
    sync_scraped_jobs()

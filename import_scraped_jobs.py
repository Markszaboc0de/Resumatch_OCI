from app import app, db, Job_Descriptions
from sqlalchemy import text
from recalculate import recalculate_jobs, refresh_all_matches

def import_scraped_jobs():
    with app.app_context():
        # Get all rows from the scraped_jobs table
        query = text("SELECT jd_id, company, title, city, country, raw_text, url FROM scraped_jobs")
        result = db.session.execute(query)
        rows = result.fetchall()
        
        if not rows:
            print("No new scraped jobs found in 'scraped_jobs' table.")
            return

        print(f"Importing {len(rows)} new scoped jobs into the main Job_Descriptions table...")
        
        for row in rows:
            # Extract data from the row
            _, company, title, city, country, raw_text, url = row
            
            # Create the main job listing
            new_job = Job_Descriptions(
                title=title,
                company=company,
                city=city,
                country=country,
                raw_text=raw_text,
                url=url,
                is_intern=False,      # Defaults
                active_status=True,   # Defaults
                is_native=False       # Defaults usually used for external links
            )
            db.session.add(new_job)

        # Clear out the scraped_jobs table so we don't import them twice next time
        db.session.execute(text("DELETE FROM scraped_jobs"))
        
        # Save to database
        db.session.commit()
        print("Successfully moved jobs to main Jobs table.")
        
        # Perform calculations for the new jobs
        print("Triggering recalculation of job embeddings...")
        recalculate_jobs()
        
        print("Triggering match refresher against all candidates...")
        refresh_all_matches()
        
        print("Import and recalculation entirely complete! The new jobs are now displayed on the website.")

if __name__ == "__main__":
    import_scraped_jobs()

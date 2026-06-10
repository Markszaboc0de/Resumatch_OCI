import csv
from app import app, db, Job_Descriptions

def export_jobs_to_csv(filename='job_board_export.csv'):
    with app.app_context():
        # Querying all jobs. You can add .filter_by(active_status=True) if you only want active jobs.
        jobs = Job_Descriptions.query.all()
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Write Header - Including all information presented on the job board
            # and used for matching (excluding raw embedding vectors to keep the CSV readable).
            writer.writerow([
                'Job ID', 
                'Title', 
                'Company', 
                'City', 
                'Country', 
                'Description (Raw Text)', 
                'URL', 
                'Is Native', 
                'Active Status', 
                'Extracted Skills',
                'Is Intern',
                'Employer ID'
            ])
            
            # Write Data
            count = 0
            for job in jobs:
                writer.writerow([
                    job.jd_id,
                    job.title,
                    job.company,
                    job.city,
                    job.country,
                    job.raw_text,
                    job.url,
                    job.is_native,
                    job.active_status,
                    job.extracted_skills,
                    job.is_intern,
                    job.employer_id
                ])
                count += 1
                
        print(f"Successfully exported {count} jobs to {filename}")

if __name__ == '__main__':
    export_jobs_to_csv()

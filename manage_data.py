import argparse
from app import app, populate_jobs, populate_resumes

def main():
    parser = argparse.ArgumentParser(description="Manage Resumatch Database")
    parser.add_argument('--reload', action='store_true', help='Clear existing data and reload from CSV files.')
    
    args = parser.parse_args()
    
    if args.reload:
        print("Reloading database from CSV files...")
        with app.app_context():
            populate_jobs(clear=True)
            populate_resumes(clear=True)
            print("Database reload complete.")
    else:
        print("No action specified. Use --reload to reset the database.")

if __name__ == "__main__":
    main()

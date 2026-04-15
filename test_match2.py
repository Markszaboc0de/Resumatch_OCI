import traceback
try:
    from app import app, db, nlp_model
    from app import clean_text
    print('Import ok')
except Exception as e:
    traceback.print_exc()

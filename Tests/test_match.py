import sys
import traceback
try:
    from app import app, db, CVs, Job_Descriptions, Users, nlp_model
    import torch
    import json
    from sentence_transformers import util
    with app.app_context():
        cv = CVs.query.filter(CVs.parsed_tokens.isnot(None)).first()
        if cv:
            try:
                vec = json.loads(cv.parsed_tokens)
                print(f"CV vector length: {len(vec)}")
            except:
                pass
        job = Job_Descriptions.query.first()
        if job:
            print(f"Job title: {job.title}")
except Exception as e:
    traceback.print_exc()

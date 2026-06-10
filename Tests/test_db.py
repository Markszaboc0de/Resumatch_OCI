import json
import traceback
from app import app, db, Job_Descriptions
with app.app_context():
    count = Job_Descriptions.query.count()
    countries = db.session.query(Job_Descriptions.country, db.func.count(Job_Descriptions.jd_id)).group_by(Job_Descriptions.country).all()
    print("Total Jobs:", count)
    print("Countries:", countries)

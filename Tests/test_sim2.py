from sentence_transformers import SentenceTransformer, util
import torch
import html
import re

nlp_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def clean_text(text):
    if not text: return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

ideal_desc = "Test job description with some details and requirements"
cleaned_desc = clean_text(ideal_desc)
ideal_embedding = nlp_model.encode(cleaned_desc, convert_to_tensor=True)

class DummyJob:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.jd_id = 1

active_jobs = [DummyJob(ideal_desc), DummyJob("Completely different job")]

job_texts = [clean_text(job.raw_text) for job in active_jobs]
job_ids = [job.jd_id for job in active_jobs]
job_embeddings = nlp_model.encode(job_texts, convert_to_tensor=True)

scores = util.cos_sim(ideal_embedding, job_embeddings)[0]
print("Scores:")
for s in scores:
    print(round(float(s)*100, 1))

EOF

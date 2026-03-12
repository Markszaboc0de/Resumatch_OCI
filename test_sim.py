from sentence_transformers import SentenceTransformer, util
import torch

nlp_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
text1 = "I am a software engineer"
text2 = "I am a software engineer"

emb1 = nlp_model.encode(text1, convert_to_tensor=True)
emb2 = nlp_model.encode([text2, "another job"], convert_to_tensor=True)

scores = util.cos_sim(emb1, emb2)[0]
for s in scores:
    print(round(float(s)*100, 1))

import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# 2) Text cleaning (regex-based)
# -----------------------------------------------------------------------------
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,2}\s*)?(?:\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})\b")
NON_ALNUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Remove noise that may bias matching:
    - Emails and phone numbers
    - Newlines/tabs and other special characters
    - Collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = EMAIL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip().lower()


# -----------------------------------------------------------------------------
# 3) Main pipeline
# -----------------------------------------------------------------------------
def main():
    # Read the external CSV containing real resumes.
    # We assume:
    # - The file is located at the given absolute path.
    # - The resume text is in the SECOND column (index 1).
    resumes_path = "/Users/mac/Desktop/Programozás/Resumatch/UpdatedResumeDataSet.csv"
    # Many European exports use ';' as a separator and may contain commas inside text.
    # Also allow for a few malformed lines without failing the whole read.
    resumes_df = pd.read_csv(
        resumes_path,
        sep=";",              # adjust if your file uses a different delimiter
        engine="python",
        on_bad_lines="skip",
    )

    # Input layout (per user):
    # - Column 0: resume ID
    # - Column 1: job category
    # - Column 2: resume text
    resume_id_col = resumes_df.columns[0]
    resume_text_col = resumes_df.columns[2]

    # Create a normalized `resume_text` column from the third column (resume text),
    # regardless of its actual name. This keeps the rest of the pipeline simple.
    resumes_df["resume_text"] = resumes_df[resume_text_col].astype(str)

    # Clean resume texts
    resumes_df["cleaned_text"] = resumes_df["resume_text"].apply(clean_text)

    # Read job descriptions:
    # - File path is provided by the user.
    # - First three columns: job_id, job_title, job_description.
    jobs_path = "/Users/mac/Desktop/Programozás/Resumatch/extracted_jobs.csv"
    # Let pandas automatically detect the delimiter here, since the job file
    # appears not to be semicolon-separated.
    jobs_df = pd.read_csv(
        jobs_path,
        sep=None,             # automatic delimiter detection
        engine="python",
        on_bad_lines="skip",
    )

    job_id_col = jobs_df.columns[0]
    job_title_col = jobs_df.columns[1]
    job_desc_col = jobs_df.columns[4]

    # Clean job descriptions
    jobs_df["cleaned_job_desc"] = jobs_df[job_desc_col].astype(str).apply(clean_text)

    # Load SBERT model (384 dimensions). If you require 1000+ dimensions,
    # project with something like PCA, but we keep the native size for accuracy.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode resume texts to dense vectors; normalize for direct cosine similarity.
    resume_vectors = model.encode(
        resumes_df["cleaned_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Encode all job descriptions at once
    job_vectors = model.encode(
        jobs_df["cleaned_job_desc"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compute cosine similarity between each job and each resume
    # similarity_matrix shape: (num_jobs, num_resumes)
    similarity_matrix = cosine_similarity(job_vectors, resume_vectors)

    # For each job, find indices of top 3 most similar resumes
    top_resume_ids_1 = []
    top_resume_ids_2 = []
    top_resume_ids_3 = []
    top_scores_1 = []
    top_scores_2 = []
    top_scores_3 = []

    for i in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[i]
        # Argsort in descending order and take top 3
        top_indices = np.argsort(-scores)[:3]

        top_ids = resumes_df[resume_id_col].iloc[top_indices].tolist()
        top_vals = scores[top_indices].tolist()

        # Pad with None if fewer than 3 resumes (edge case)
        while len(top_ids) < 3:
            top_ids.append(None)
            top_vals.append(None)

        top_resume_ids_1.append(top_ids[0])
        top_resume_ids_2.append(top_ids[1])
        top_resume_ids_3.append(top_ids[2])
        top_scores_1.append(top_vals[0])
        top_scores_2.append(top_vals[1])
        top_scores_3.append(top_vals[2])

    # Build output DataFrame: first three columns are job id, title, description,
    # then three columns for the IDs of the top three matching resumes,
    # followed by their corresponding match scores.
    output_df = jobs_df[[job_id_col, job_title_col, job_desc_col]].copy()
    output_df["top_resume_id_1"] = top_resume_ids_1
    output_df["top_resume_id_2"] = top_resume_ids_2
    output_df["top_resume_id_3"] = top_resume_ids_3
    output_df["top_resume_score_1"] = top_scores_1
    output_df["top_resume_score_2"] = top_scores_2
    output_df["top_resume_score_3"] = top_scores_3

    # Save in European CSV format (semicolon-separated)
    output_df.to_csv("job_to_top_resumes.csv", index=False, sep=";")
    print("\nJob-to-resume mapping saved to job_to_top_resumes.csv (semicolon-separated)")


if __name__ == "__main__":
    main()


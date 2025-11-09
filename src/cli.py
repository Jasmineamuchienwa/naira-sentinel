import argparse, os, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def scan_csv(path, vec_path="models/vectorizer.joblib", model_path="models/sentinel_model.joblib"):
    if not os.path.exists(path):
        print("File not found:", path); return
    df = pd.read_csv(path)
    texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)
    vec = joblib.load(vec_path)
    clf = joblib.load(model_path)
    X = vec.transform(texts)
    probs = clf.predict_proba(X)[:,1]
    for i, (subj, p) in enumerate(zip(df["subject"], probs)):
        tag = "PHISH" if p >= 0.5 else "OK"
        print(f"{i:03d} | {p:0.2f} | {tag} | {subj}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="?", default="scan")
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    scan_csv(args.file)

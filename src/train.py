import os, argparse, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import joblib
from dataset_generator import generate_csv

def clean_text(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r'https?://\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def main(n_samples=2000):
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    data_path = "samples/sample_emails.csv"
    if not os.path.exists(data_path):
        generate_csv(path=data_path, n=n_samples)

    df = pd.read_csv(data_path)
    texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)
    y = df["label"].values

    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(texts)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    joblib.dump(vec, "models/vectorizer.joblib")
    joblib.dump(clf, "models/sentinel_model.joblib")

    probs = clf.predict_proba(Xte)[:,1]
    preds = (probs >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary", zero_division=0)
    auc = roc_auc_score(yte, probs)

    with open("artifacts/eval.txt","w") as f:
        f.write(f"precision: {prec}\nrecall: {rec}\nf1: {f1}\nauc: {auc}\n")
    print("Training done. Eval:", {"precision": prec, "recall": rec, "f1": f1, "auc": auc})

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=2000)
    args = p.parse_args()
    main(n_samples=args.n_samples)

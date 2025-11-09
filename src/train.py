# FILE: src/train.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

try:
    from src.utils import clean_text
    from src.features import MetaFeatureizer
except ModuleNotFoundError:
    from utils import clean_text
    from features import MetaFeatureizer

PIPELINE_OUT = Path("models/phishing_clf.joblib")

def _ensure_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns:
        return df
    if "subject" in df.columns and "body" in df.columns:
        df = df.copy()
        df["text"] = df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
        return df
    raise ValueError("CSV needs a 'text' column or both 'subject' and 'body'.")

def train(csv_path: Path | str, model_out: Path | str = PIPELINE_OUT) -> dict:
    csv_path = Path(csv_path)
    model_out = Path(model_out)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = _ensure_text_column(df)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0/1).")

    X = df["text"].astype(str).map(clean_text)
    y = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True)),
        ("meta", MetaFeatureizer()),
    ])

    pipe = Pipeline([
        ("feats", feats),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ])

    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_te, proba)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_out)

    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc),
            "n_test": int(len(y_te)), "model_path": str(model_out)}

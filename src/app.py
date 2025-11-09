# FILE: src/app.py
from __future__ import annotations
import io
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import make_pipeline

# Local modules
# Both import styles so this works whether run from root or src
try:
    from src.utils import clean_text
    from src.explain import simple_reason, find_region_hits
except ModuleNotFoundError:
    from utils import clean_text
    from explain import simple_reason, find_region_hits

# Model artifacts
PIPELINE_PATH = Path("models/phishing_clf.joblib")
VECTORIZER_PATH = Path("models/vectorizer.joblib")
MODEL_PATH = Path("models/sentinel_model.joblib")

@st.cache_resource
def load_pipeline():
    if PIPELINE_PATH.exists():
        return load(PIPELINE_PATH)
    if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
        vec = load(VECTORIZER_PATH)
        clf = load(MODEL_PATH)
        return make_pipeline(vec, clf)
    raise FileNotFoundError(
        "No model found. Train first. Expected one of:\n"
        f"- {PIPELINE_PATH}\n- {VECTORIZER_PATH} + {MODEL_PATH}"
    )

def score_texts(texts: List[str], threshold: float, region: str) -> List[Dict]:
    pipe = load_pipeline()
    probs = pipe.predict_proba([clean_text(t) for t in texts])[:, 1]
    rows: List[Dict] = []
    for t, p in zip(texts, probs):
        verdict = "PHISH" if p >= threshold else "OK"
        rows.append({
            "text": t,
            "prob": float(p),
            "verdict": verdict,
            "reason": simple_reason(t),
            "region_hits": ", ".join(find_region_hits(t, region)) or "-"
        })
    return rows

def coerce_text_column(df: pd.DataFrame) -> pd.DataFrame:
    # prefer 'text'; else 'subject' + 'body'
    if "text" in df.columns:
        return df
    subj = df["subject"] if "subject" in df.columns else ""
    body = df["body"] if "body" in df.columns else ""
    df = df.copy()
    if isinstance(subj, pd.Series) and isinstance(body, pd.Series):
        df["text"] = subj.fillna("").astype(str) + " " + body.fillna("").astype(str)
    else:
        raise ValueError("CSV needs a 'text' column or both 'subject' and 'body'.")
    return df

# ─── UI ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naira Sentinel — Global",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Naira Sentinel — Global")
st.caption("Local phishing detector with region-aware explanations. Nothing leaves your device.")

with st.sidebar:
    st.subheader("Settings")
    region = st.selectbox("Region context", ["global", "africa", "uk", "us"], index=0)
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
    preview_n = st.slider("Preview rows", 5, 50, 20)
    st.markdown("---")
    st.write("**Input mode**")
    mode = st.radio("Choose input", ["Single message", "CSV upload"], horizontal=True)

# Single message mode
if mode == "Single message":
    txt = st.text_area("Paste an email/message to scan", height=160,
                       placeholder="e.g., Urgent: Your account will be locked. Verify at http://secure-login…")
    if st.button("Scan message", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("Please paste a message first.")
        else:
            rows = score_texts([txt], threshold, region)
            r = rows[0]
            verdict_badge = "🔴 PHISH" if r["verdict"] == "PHISH" else "🟢 OK"
            st.metric("Verdict", verdict_badge, delta=f"p={r['prob']:.3f}")
            st.write("**Reason:**", r["reason"])
            st.write("**Region hits:**", r["region_hits"])

# CSV mode
else:
    st.write("Upload a CSV with a `text` column (or `subject` + `body`).")
    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            df = coerce_text_column(df)
        except Exception as e:
            st.error(f"Could not read your CSV: {e}")
            st.stop()

        st.success(f"Loaded {len(df)} rows.")
        if st.button("Scan CSV", type="primary", use_container_width=True):
            rows = score_texts(df["text"].astype(str).tolist(), threshold, region)
            out = pd.DataFrame(rows)
            # Summary
            total = len(out)
            phish = (out["verdict"] == "PHISH").sum()
            ok = total - phish
            left, right = st.columns(2)
            with left:
                st.metric("Total scanned", total)
                st.metric("Phish flagged", f"{phish} ({phish/total*100:.1f}%)")
            with right:
                st.metric("Safe", f"{ok} ({ok/total*100:.1f}%)")
                st.metric("Avg probability", f"{out['prob'].mean():.3f}")

            # Charts
            st.subheader("Verdicts")
            vc = out["verdict"].value_counts()
            st.bar_chart(vc)

            st.subheader("Probability distribution")
            st.line_chart(out["prob"].sort_values(ignore_index=True))

            st.subheader("Preview")
            st.dataframe(out.head(preview_n))

            # Download results
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="naira_sentinel_scanned.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Lightweight HTML export (inline)
            html = io.StringIO()
            html.write("<html><head><meta charset='utf-8'><title>Naira Sentinel Report</title></head><body>")
            html.write(f"<h1>Naira Sentinel — Global</h1><p>Total: {total} • Phish: {phish} • Safe: {ok}</p>")
            html.write(out.head(200).to_html(index=False, escape=True))
            html.write("</body></html>")
            st.download_button(
                "Download mini HTML report",
                data=html.getvalue().encode("utf-8"),
                file_name="naira_sentinel_report.html",
                mime="text/html",
                use_container_width=True
            )

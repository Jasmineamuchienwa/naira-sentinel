# FILE: src/features.py adds smart signals like counting links checking for all caps and suspicious websites
from __future__ import annotations
import re
import numpy as np
from typing import Iterable
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"https?://[^\s)>\]]+", re.I)
SUSP_TLDS = {".zip", ".mov", ".xyz", ".top", ".lol", ".ru", ".cn", ".gq", ".tk", ".ml", ".cf"}

def _url_stats(text: str):
    urls = URL_RE.findall(text or "")
    n = len(urls)
    if n == 0:
        return 0, 0.0, 0
    lens = [len(u) for u in urls]
    avg_len = float(np.mean(lens))
    susp = any(any(u.lower().endswith(t) for t in SUSP_TLDS) for u in urls)
    return n, avg_len, int(susp)

def _caps_ratio(text: str):
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

def _digit_ratio(text: str):
    t = text or ""
    return sum(1 for c in t if c.isdigit()) / max(1, len(t))

class MetaFeatureizer(BaseEstimator, TransformerMixin):
    """
    Turns raw text -> numeric features:
      [url_count, url_avg_len, suspicious_tld, exclamations, caps_ratio, digit_ratio, http_count]
    Returns sparse matrix for scikit-learn pipelines.
    """
    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        rows = []
        for t in X:
            url_cnt, url_avg, susp = _url_stats(t)
            exclam = (t or "").count("!")
            caps_r = _caps_ratio(t)
            digit_r = _digit_ratio(t)
            http_cnt = (t or "").lower().count("http")
            rows.append([url_cnt, url_avg, susp, exclam, caps_r, digit_r, http_cnt])
        return csr_matrix(np.asarray(rows, dtype=float))

# FILE: src/utils.py
import re

def clean_text(text: str) -> str:
    """Basic text cleaner for phishing detection."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9\s@_%-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

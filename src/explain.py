# FILE: src/explain.py
from __future__ import annotations
import re
from typing import Dict, List

# Region keyword catalog (expand anytime)
REGION_KEYWORDS: Dict[str, List[str]] = {
    "global": [
        "verify", "urgent", "reset password", "click here", "confirm account",
        "update billing", "invoice", "attachment", "password", "login", "credential"
    ],
    "africa": [
        "UBA", "GTBank", "Access Bank", "First Bank", "BVN", "MTN", "Glo", "Airtel", "CBN", "NIN"
    ],
    "uk": [
        "HMRC", "Royal Mail", "NHS", "DVLA", "TV Licence", "Council Tax", "Bank of Scotland", "Monzo", "Barclays"
    ],
    "us": [
        "IRS", "USPS", "Apple ID", "PayPal", "Chase", "Wells Fargo", "Social Security", "Venmo", "Amazon refund"
    ],
}

SUSPICIOUS_URL_RE = re.compile(r"https?://[^\s]+", re.I)

def find_region_hits(text: str, region: str) -> List[str]:
    region = (region or "global").lower()
    buckets = ["global"]
    if region in REGION_KEYWORDS and region != "global":
        buckets.append(region)
    words = set()
    low = text.lower()
    for b in buckets:
        for kw in REGION_KEYWORDS[b]:
            if kw.lower() in low:
                words.add(kw)
    return sorted(words)

def simple_reason(text: str) -> str:
    t = text.lower()
    reasons = []
    if any(w in t for w in ["verify", "verification", "confirm", "validate"]):
        reasons.append("asks to verify/confirm account")
    if any(w in t for w in ["urgent", "immediately", "now", "action required"]):
        reasons.append("uses urgent or coercive language")
    if SUSPICIOUS_URL_RE.search(t):
        reasons.append("contains hyperlink(s)")
    if any(w in t for w in ["password", "otp", "pin", "credentials", "login"]):
        reasons.append("requests credentials or codes")
    if any(w in t for w in ["invoice", "billing", "payment", "refund"]):
        reasons.append("mentions payment/refund/invoice")
    if not reasons:
        reasons.append("pattern match / model confidence")
    return "; ".join(reasons)

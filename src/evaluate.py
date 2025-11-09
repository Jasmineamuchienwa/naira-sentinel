
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import textwrap

RAW_CSV = Path("samples/sample_emails.csv")
SCAN_CSV = Path("samples/sample_emails_scanned.csv")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True, parents=True)

# Load both files
raw = pd.read_csv(RAW_CSV)
scanned = pd.read_csv(SCAN_CSV)

# Sanity checks
if "verdict" not in scanned.columns or "prob" not in scanned.columns:
    raise SystemExit("scanned CSV must have 'verdict' and 'prob' columns.")

# If label exists in the raw data, attach it to the scanned results (row-aligned)
if "label" in raw.columns and len(raw) == len(scanned):
    scanned["label"] = raw["label"].values
else:
    print("⚠️  No labels found (or length mismatch) — metrics will be skipped.")
    scanned.to_csv(REPORTS / "scanned_with_labels.csv", index=False)
    print(f"Saved: {REPORTS / 'scanned_with_labels.csv'}")
    raise SystemExit(0)

# Convert verdict -> 1/0
scanned["pred"] = (scanned["verdict"].str.upper() == "PHISH").astype(int)
y_true = scanned["label"].astype(int)
y_pred = scanned["pred"].astype(int)

# Metrics
report = classification_report(y_true, y_pred, target_names=["SAFE(0)", "PHISH(1)"])
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()

# Save a quick markdown report
md = f"""# Naira Sentinel — Evaluation Report

**Total samples:** {len(scanned)}

## Confusion Matrix (labels: 0=SAFE, 1=PHISH)
- TN (correct safe):  {tn}
- FP (false phish):   {fp}
- FN (missed phish):  {fn}
- TP (correct phish): {tp}

## Classification Report
"""
(REPORTS / "evaluation_report.md").write_text(md)
print("✅ Saved:", REPORTS / "evaluation_report.md")

# Save a simple confusion matrix bar chart
plt.figure(figsize=(6,4))
plt.bar(["TN","FP","FN","TP"], [tn, fp, fn, tp])
plt.title("Confusion Matrix Counts")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(REPORTS / "confusion_counts.png", dpi=160)
print("✅ Saved:", REPORTS / "confusion_counts.png")

# Also save the merged CSV for reference
out_csv = REPORTS / "scanned_with_labels.csv"
scanned.to_csv(out_csv, index=False)
print("✅ Saved:", out_csv)

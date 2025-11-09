# FILE: src/report_html.py
import pandas as pd
from pathlib import Path
from datetime import datetime

SCANNED = Path("samples/sample_emails_scanned.csv")
RAW = Path("samples/sample_emails.csv")
OUTDIR = Path("reports")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTHTML = OUTDIR / "naira_sentinel_report.html"

if not SCANNED.exists():
    raise SystemExit(f"Missing {SCANNED}. Run scan-csv first.")

scanned = pd.read_csv(SCANNED)

# summary
total = len(scanned)
phish = (scanned["verdict"].str.upper() == "PHISH").sum()
ok = total - phish
avg_prob = scanned["prob"].mean() if "prob" in scanned.columns else float("nan")

# attach labels if available and same length
accuracy_block = ""
if RAW.exists():
    raw = pd.read_csv(RAW)
    if "label" in raw.columns and len(raw) == len(scanned):
        y_true = raw["label"].astype(int)
        y_pred = (scanned["verdict"].str.upper() == "PHISH").astype(int)
        acc = (y_true == y_pred).mean() * 100.0
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        accuracy_block = f"""
        <h3>Evaluation (with labels)</h3>
        <ul>
          <li>Accuracy: {acc:.1f}%</li>
          <li>TP: {tp} &nbsp; TN: {tn} &nbsp; FP: {fp} &nbsp; FN: {fn}</li>
        </ul>
        """

# top flagged
top = scanned.sort_values("prob", ascending=False).head(25)
top_tbl = top[["text", "prob", "verdict"]].copy()
top_tbl["prob"] = top_tbl["prob"].map(lambda x: f"{x:.3f}")
top_html = top_tbl.to_html(index=False, escape=True)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Naira Sentinel — Scan Report</title>
<style>
 body {{ font-family: -apple-system, Inter, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
 .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; }}
 h1 {{ margin: 0 0 4px; }}
 .muted {{ color: #666; font-size: 0.95rem; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: left; }}
 th {{ background: #fafafa; }}
 .pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
 .ok {{ background:#e8f7ee; color:#0f7b3e; }}
 .phish {{ background:#fde8ea; color:#b42318; }}
</style>
</head>
<body>
  <div class="card">
    <h1>Naira Sentinel — Scan Report</h1>
    <div class="muted">Generated: {datetime.utcnow().isoformat()}Z</div>
  </div>

  <div class="card">
    <h2>Summary</h2>
    <ul>
      <li>Total emails scanned: <b>{total}</b></li>
      <li>Flagged as phishing: <b>{phish}</b> ({phish/total*100:.1f}%)</li>
      <li>Safe: <b>{ok}</b> ({ok/total*100:.1f}%)</li>
      <li>Average phishing probability: <b>{avg_prob:.3f}</b></li>
    </ul>
    {accuracy_block}
  </div>

  <div class="card">
    <h2>Top 25 Flagged by Probability</h2>
    {top_html}
  </div>

  <div class="muted">Naira Sentinel • local CLI phishing detector</div>
</body>
</html>
"""

OUTHTML.write_text(html, encoding="utf-8")
print(f"✅ HTML report written to {OUTHTML}")

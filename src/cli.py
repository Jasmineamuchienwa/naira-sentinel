from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import List, Dict

from joblib import load
from sklearn.pipeline import make_pipeline

# Flexible imports: works as "python -m src.cli" and "python src/cli.py"
try:
    from src.utils import clean_text
    from src.dataset_generator import generate_csv
except ModuleNotFoundError:
    from utils import clean_text
    from dataset_generator import generate_csv

# Preferred single-pipeline (if you trained with src/train.py writing one file)
PIPELINE_PATH = Path("models/phishing_clf.joblib")

# Fallback: separate artifacts (what you currently have)
VECTORIZER_PATH = Path("models/vectorizer.joblib")
MODEL_PATH      = Path("models/sentinel_model.joblib")

def load_model_pipeline():
    """Load a single saved pipeline or build one from vectorizer+classifier."""
    if PIPELINE_PATH.exists():
        return load(PIPELINE_PATH)

    if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
        vec = load(VECTORIZER_PATH)
        clf = load(MODEL_PATH)
        return make_pipeline(vec, clf)

    print(
        "No model found.\n"
        f"Checked:\n - {PIPELINE_PATH}\n - {VECTORIZER_PATH} + {MODEL_PATH}\n"
        "Train first or ensure one of the above exists.",
        file=sys.stderr,
    )
    sys.exit(1)

def score_texts(texts: List[str], threshold: float) -> List[Dict]:
    pipe = load_model_pipeline()
    probs = pipe.predict_proba([clean_text(t) for t in texts])[:, 1]
    out = []
    for t, p in zip(texts, probs):
        verdict = "PHISH" if p >= threshold else "OK"
        out.append({"text": t, "prob": float(p), "verdict": verdict})
    return out

# ---------- pretty print helper ----------
def pretty_print_rows(rows):
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        table = Table(title="Naira Sentinel — Scan Results", box=box.SIMPLE_HEAVY)
        table.add_column("Verdict", justify="center")
        table.add_column("Prob", justify="right")
        table.add_column("Text (truncated)")
        for r in rows:
            verdict = "[bold red]PHISH[/]" if r["verdict"] == "PHISH" else "[green]OK[/]"
            table.add_row(verdict, f"{r['prob']:.3f}", r["text"][:100])
        console.print(table)
    except Exception:
        # fallback plain
        for r in rows[:10]:
            print(f"[{r['verdict']}] p={r['prob']:.3f} :: {r['text'][:100]}")

# ----------------- subcommand handlers -----------------
def cmd_generate(args: argparse.Namespace):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = generate_csv(out, n=args.n, phish_ratio=args.ratio)
    print(f"Generated: {p} (n={args.n}, phish_ratio={args.ratio})")

def cmd_train(args: argparse.Namespace):
    # lazy import to speed up CLI when just scanning
    try:
        from src.train import train
    except ModuleNotFoundError:
        from train import train
    csv_path = Path(args.csv)
    model_out = PIPELINE_PATH  # always write single-file pipeline when training via CLI
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics = train(csv_path, model_out)
    print("Training done. Eval:", metrics)

def cmd_scan_text(args: argparse.Namespace):
    row = score_texts([args.text], threshold=args.threshold)[0]
    if args.pretty:
        pretty_print_rows([row])
    else:
        print(json.dumps(row, indent=2))

def cmd_scan_csv(args: argparse.Namespace):
    inp = Path(args.csv)
    if not inp.exists():
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    with inp.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "text" not in r.fieldnames:
            print("CSV must have a 'text' column.", file=sys.stderr)
            sys.exit(1)
        texts = [row["text"] for row in r]

    rows = score_texts(texts, threshold=args.threshold)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "prob", "verdict"])
            for rrow in rows:
                w.writerow([rrow["text"], rrow["prob"], rrow["verdict"]])
        print(f"Report saved: {out}")

    # Always show a preview
    if args.pretty:
        pretty_print_rows(rows[:20])
    else:
        for rrow in rows[:10]:
            print(f"[{rrow['verdict']}] p={rrow['prob']:.3f} :: {rrow['text'][:100]}")

# ----------------- argparse wiring -----------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nairasentinel", description="Local CLI phishing detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic dataset")
    g.add_argument("--out", default="samples/sample_emails.csv")
    g.add_argument("-n", type=int, default=1000)
    g.add_argument("--ratio", type=float, default=0.5, help="phishing proportion")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Train a model on a CSV")
    t.add_argument("--csv", default="samples/sample_emails.csv")
    t.set_defaults(func=cmd_train)

    s1 = sub.add_parser("scan-text", help="Score a single text")
    s1.add_argument("text")
    s1.add_argument("--threshold", type=float, default=0.5)
    s1.add_argument("--pretty", action="store_true")
    s1.set_defaults(func=cmd_scan_text)

    s2 = sub.add_parser("scan-csv", help="Score all rows in a CSV (must have 'text' column)")
    s2.add_argument("csv")
    s2.add_argument("--threshold", type=float, default=0.5)
    s2.add_argument("--out", default="", help="Write results to CSV if set")
    s2.add_argument("--pretty", action="store_true")
    s2.set_defaults(func=cmd_scan_csv)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

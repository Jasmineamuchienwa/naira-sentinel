import pandas as pd

# Load results
df = pd.read_csv("samples/sample_emails_scanned.csv")

# Ensure required columns exist
if not {"text", "prob", "verdict"}.issubset(df.columns):
    raise ValueError("Your CSV must have 'text', 'prob', and 'verdict' columns.")

# Total and proportions
total = len(df)
phish = (df["verdict"] == "PHISH").sum()
ok = total - phish
phish_pct = (phish / total) * 100

# Confidence stats
avg_prob = df["prob"].mean()
high_conf_phish = ((df["verdict"] == "PHISH") & (df["prob"] > 0.9)).sum()

# Print summary
print("\n🧾 Naira Sentinel — Summary Report")
print("────────────────────────────────────────────")
print(f"Total emails scanned:        {total}")
print(f"Phishing detected:            {phish} ({phish_pct:.1f}%)")
print(f"Safe emails:                  {ok} ({100 - phish_pct:.1f}%)")
print(f"Average phishing probability: {avg_prob:.3f}")
print(f"High-confidence PHISH hits:   {high_conf_phish}")
print("────────────────────────────────────────────")

# Optional: show sample flagged messages
flagged = df[df["verdict"] == "PHISH"].head(5)
if not flagged.empty:
    print("\n⚠️  Sample flagged messages:")
    for _, row in flagged.iterrows():
        print(f" - {row['text'][:90]}... (p={row['prob']:.2f})")
else:
    print("\n✅ No phishing messages detected.")

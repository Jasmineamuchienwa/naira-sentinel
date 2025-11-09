import pandas as pd
import matplotlib.pyplot as plt

# Load scanned results
df = pd.read_csv("samples/sample_emails_scanned.csv")

# Count verdicts
counts = df["verdict"].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#6BCB77", "#FF6B6B"],
)
plt.title("Phishing Detection Results")
plt.show()

# Create a histogram of probabilities
plt.figure(figsize=(8, 4))
plt.hist(df["prob"], bins=20, color="#4D96FF", edgecolor="black")
plt.title("Probability Distribution of Predictions")
plt.xlabel("Phishing Probability")
plt.ylabel("Frequency")
plt.show()

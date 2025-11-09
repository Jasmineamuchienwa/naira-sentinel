# 🛡️ The Naira Sentinel Project

A command-line phishing and malware detection tool built with Python, scikit-learn, and synthetic data.  
Developed to simulate real-world phishing detection in a safe and educational way.

---

## 🚀 Features
- Generates fake phishing and normal emails
- Trains a machine-learning model to detect threats
- Scans text data via CLI and flags suspicious messages
- Built using TF-IDF + Logistic Regression
- Uses synthetic, ethical data only (no real emails)

---

## 💻 How to Run
```bash
# Train the model
python src/train.py --n-samples 1000

# Scan synthetic emails
python src/cli.py --file samples/sample_emails.csv
```

---

## 🧠 Example Output
```
000 | 0.87 | PHISH | Action Required: GTBank Verification
001 | 0.09 | OK    | Team meeting notes
```

---

## 📚 About
This project was built as part of my personal cybersecurity portfolio.  
It showcases practical defensive concepts like phishing detection, data analysis, and ethical model design.

**Languages:** Python  
**Libraries:** pandas, scikit-learn, joblib, tqdm, rich  
**Role:** Developer & Cybersecurity Student  

---

## 🧩 Future Goals
- Add a Flask-based dashboard  
- Introduce NLP for smarter classification  
- Integrate live email scanning simulation  

---

### 🩵 Author
**Jasmine Amuchienwa**  
Cybersecurity student at Manchester Metropolitan University  
[LinkedIn](https://www.linkedin.com/in/fareedah-phillips-0b54a22a3/) | [Portfolio](#) | [GitHub](https://github.com/Jasmineamuchienwa)


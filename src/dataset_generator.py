import random, csv, os
from datetime import datetime

PHISHING_TEMPLATES = [
    "Dear {name}, we detected unusual activity on your {service} account. Verify: {link}",
    "Your {service} payment failed. Update billing: {link}",
    "Congratulations {name}! You won a prize. Claim now: {link}",
    "{name}, important security notice for your {service} — click {link} to resolve.",
]
BENIGN_TEMPLATES = [
    "Hey {name}, are we still on for tomorrow at 6pm?",
    "Meeting notes attached. See you in the seminar.",
    "Can you review the attached doc and approve?",
    "Lunch on campus? I'm free after 2pm.",
]
SERVICES = ["Bank of Naira","UniPortal","PayFastNG","AzumiPay","GigaMail"]
DOMAINS  = ["naira-bank.ng","unimmu.edu.ng","payfast.ng","azumi.io","giga-mail.com"]
NAMES    = ["Aisha","Chinedu","Emeka","Ngozi","Olu","Fatima","John","Mary"]

os.makedirs("samples", exist_ok=True)

def gen_link(domain):
    if random.random() < 0.6:
        return f"https://{domain}/secure/login?id={random.randint(1000,9999)}"
    return f"http://{domain.replace('.', '-')}.verify-account.ng/{random.randint(100,999)}"

def generate_email(is_phish: bool):
    name, service, domain, link = random.choice(NAMES), random.choice(SERVICES), random.choice(DOMAINS), gen_link(random.choice(DOMAINS))
    if is_phish:
        body = random.choice(PHISHING_TEMPLATES).format(name=name, service=service, link=link)
        subject = random.choice([f"Action Required: {service}", f"{service} security alert", f"Payment issue on {service}"])
        sender = f"no-reply@{domain}"
    else:
        body = random.choice(BENIGN_TEMPLATES).format(name=name)
        subject = random.choice(["Lunch plan","Meeting notes","Project update","Reminder"])
        sender = f"{name.lower()}@student.unimmu.edu.ng"
    return {"subject": subject, "body": body, "from": sender, "to": "you@example.com", "date": datetime.utcnow().isoformat(), "label": int(is_phish)}

def generate_csv(path="samples/sample_emails.csv", n=1000, phish_ratio=0.3):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject","body","from","to","date","label"])
        w.writeheader()
        for _ in range(n):
            w.writerow(generate_email(random.random() < phish_ratio))
    print(f"Generated {n} samples at {path}")

if __name__ == "__main__":
    generate_csv(n=1000)

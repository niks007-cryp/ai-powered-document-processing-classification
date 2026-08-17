import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TEMPLATES = {
    "invoice": [
        "Invoice {id} from {vendor}. Invoice date {date}. Total amount INR {amount}. Payment due {due}. Customer {customer}.",
        "Tax invoice {id}. Supplier {vendor}. Amount due INR {amount}. Invoice date {date}. Due date {due}. Bill to {customer}.",
    ],
    "loan_application": [
        "Loan application {id}. Applicant {customer}. Requested amount INR {amount}. Employment {job}. Application date {date}.",
        "Credit application {id} for {customer}. Requested loan INR {amount}. Employment status {job}. Submitted {date}.",
    ],
    "employment_letter": [
        "Employment letter for {customer}. Employer {vendor}. Joining date {date}. Annual salary INR {amount}. Role {job}.",
        "Offer letter issued by {vendor} to {customer}. Start date {date}. Compensation INR {amount}. Position {job}.",
    ],
    "complaint": [
        "Customer complaint {id}. Customer {customer}. Issue reported on {date}: payment failed and account was charged. Reference amount INR {amount}.",
        "Service complaint {id} from {customer}. Date {date}. Reported issue: transaction failed. Amount INR {amount}.",
    ],
}


def make_dataset(n_per_class=120, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    names = ["Aarav Mehta", "Priya Shah", "Rohan Kulkarni", "Neha Patil", "Vikram Rao"]
    vendors = ["Acme Supplies", "Nimbus Finance", "Vertex Services", "Bright Retail"]
    jobs = ["Analyst", "Manager", "Engineer", "Associate"]
    for label, templates in TEMPLATES.items():
        for i in range(n_per_class):
            rows.append((templates[i % len(templates)].format(
                id=f"D{i+1:04d}", customer=rng.choice(names), vendor=rng.choice(vendors),
                date=f"2026-{rng.integers(1,13):02d}-{rng.integers(1,28):02d}",
                due=f"2026-{rng.integers(1,13):02d}-{rng.integers(1,28):02d}",
                amount=f"{rng.integers(5000,200000):,}", job=rng.choice(jobs)), label))
    rng.shuffle(rows)
    return rows


def train(seed=42):
    rows = make_dataset(seed=seed)
    x, y = [r[0] for r in rows], [r[1] for r in rows]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    return model, {"accuracy": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro"), "n_test": len(yte)}


def extract_fields(text):
    def find(pattern):
        m = re.search(pattern, text, re.I)
        return m.group(1).strip() if m else None
    return {
        "document_id": find(r"(?:Invoice|application|complaint|letter|document)\s+(?:ID\s*)?([A-Z]?\d{3,})"),
        "customer": find(r"(?:Bill to|Customer|Applicant)\s*[:\-]?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,2})"),
        "amount_inr": find(r"(?:INR|Rs\.?)\s*([0-9,]+)"),
        "date": find(r"(?:date|issued on|submitted|Joining date|Start date)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})"),
    }

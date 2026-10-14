"""Training pipeline for the Document Classification model.

Trains a TF-IDF vectorizer + Logistic Regression classifier on labeled domain documents
and serializes artifacts to the models directory.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import CLASSIFIER_MODEL_PATH, MODELS_DIR, VECTORIZER_MODEL_PATH


def get_default_training_corpus() -> Tuple[List[str], List[str]]:
    """Return labeled training corpus spanning document categories.

    Returns:
        Tuple[List[str], List[str]]: (texts, labels).
    """
    corpus = [
        # Medical Records
        (
            "Medical Prescription. Patient Name: John Smith. Rx: Amoxicillin 500mg capsules twice daily for 7 days. Diagnosis: Acute bacterial bronchitis. Dr. Elizabeth Warren, MD.",
            "Medical Record",
        ),
        (
            "Clinic Consultation Note. Patient complains of severe migraine and nausea. Prescribed Sumatriptan 50mg as needed. Follow up in two weeks if symptoms persist.",
            "Medical Record",
        ),
        (
            "Hospital Discharge Summary. Patient admitted for appendectomy. Vitals stable. Prescribed Ibuprofen 400mg every 6 hours for postoperative pain. Blood pressure 120/80.",
            "Medical Record",
        ),
        (
            "Laboratory diagnostic report. Fasting blood sugar 110 mg/dL, HbA1c 6.2%. Patient advised diet modification and regular exercise. Prescribed Metformin 500mg daily.",
            "Medical Record",
        ),
        # Invoices / Receipts
        (
            "Tax Invoice # INV-2024-984. Total amount due: $1,250.00. Payment due date: 15/11/2024. Description: Consulting services, Qty 20 hours at $62.50 per hour. Subtotal $1250.00.",
            "Invoice / Receipt",
        ),
        (
            "Official Sales Receipt. Date: 2024-10-01. Customer: Acme Corp. Item 1: Office Supplies $45.00. Item 2: Printer Toner $85.50. Sales Tax 8%: $10.44. Total Paid: $140.94. Thank you for your business.",
            "Invoice / Receipt",
        ),
        (
            "Commercial Invoice. Bill To: Global Trading Ltd. Due Date: 2024-12-01. Items: Industrial parts. Subtotal: $4,500.00. Shipping: $120.00. VAT: $450.00. Total Balance: $5,070.00.",
            "Invoice / Receipt",
        ),
        (
            "Payment Receipt voucher. Paid to: Express Logistics. Amount: $320.50 USD. Method: Bank transfer. Invoice Ref: 44109. Status: Settled in full.",
            "Invoice / Receipt",
        ),
        # Application Forms
        (
            "Official Employment Application Form. Full Legal Name: Robert Miller. Position Applied For: Senior Systems Analyst. Date of Birth: 1988-04-12. Citizenship: USA. Signature: R. Miller.",
            "Application Form",
        ),
        (
            "Passport and Visa Application Form. Application Reference: APP-881290. Applicant Details: Mary Jane Watson. Residential Address: 124 Elm Street, Springfield. Verified signature attached.",
            "Application Form",
        ),
        (
            "University Admission Application. Program: Master of Science in Computer Science. Semester: Fall 2025. Applicant ID: 90218. Declaration: I certify that all statements made herein are true and accurate.",
            "Application Form",
        ),
        (
            "Membership Application Form. Organization: Regional Athletic Club. Primary Contact: David Brown. Phone: 555-0144. Applicant signature and date 2024-09-18.",
            "Application Form",
        ),
        # Handwritten Notes / Letters
        (
            "Dear Sarah, Thank you so much for hosting us last weekend. We had a wonderful time catching up and reminiscing about old memories. Looking forward to seeing you soon! Warm regards, Linda.",
            "Handwritten Note / Letter",
        ),
        (
            "Meeting Notes: Quick action items from Monday sync. 1. Follow up with client regarding project timeline. 2. Prepare slide deck for board meeting. 3. Call Michael about budget adjustments.",
            "Handwritten Note / Letter",
        ),
        (
            "Personal reminder note: Don't forget to buy groceries tonight - milk, eggs, sourdough bread, and coffee beans. Also drop off dry cleaning before 6 PM.",
            "Handwritten Note / Letter",
        ),
        (
            "Dear Professor Adams, I am writing to express my gratitude for your mentorship throughout the past semester. Your feedback on my research paper was invaluable. Sincerely, Alex Turner.",
            "Handwritten Note / Letter",
        ),
    ]
    texts = [item[0] for item in corpus]
    labels = [item[1] for item in corpus]
    return texts, labels


def train_and_save(
    classifier_path: Path = CLASSIFIER_MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_MODEL_PATH,
) -> None:
    """Train TF-IDF + Logistic Regression and serialize fitted models to disk.

    Args:
        classifier_path (Path): Destination for Logistic Regression model.
        vectorizer_path (Path): Destination for TfidfVectorizer.
    """
    texts, labels = get_default_training_corpus()

    print("[Classifier Training] Fitting TF-IDF Vectorizer (n-grams: 1-2)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000)
    X = vectorizer.fit_transform(texts)

    print("[Classifier Training] Fitting Logistic Regression model...")
    classifier = LogisticRegression(max_iter=500, C=5.0, random_state=42)
    classifier.fit(X, labels)

    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"[Classifier Training] Models successfully saved to:\n  {classifier_path}\n  {vectorizer_path}")


if __name__ == "__main__":
    train_and_save()

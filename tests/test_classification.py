"""Tests for Document Classification module."""

import pytest
from src.classification.classifier import DocumentClassifier
from src.classification.train_classifier import train_and_save


def test_untrained_classifier_reports_not_trained(tmp_path):
    clf = DocumentClassifier(
        classifier_path=tmp_path / "nonexistent_clf.joblib",
        vectorizer_path=tmp_path / "nonexistent_vec.joblib",
    )
    res = clf.predict("Prescription for Amoxicillin")
    assert not res.is_trained
    assert res.predicted_class == "General / Unclassified"


def test_classifier_training_and_inference(tmp_path):
    clf_path = tmp_path / "test_clf.joblib"
    vec_path = tmp_path / "test_vec.joblib"

    train_and_save(classifier_path=clf_path, vectorizer_path=vec_path)

    clf = DocumentClassifier(classifier_path=clf_path, vectorizer_path=vec_path)
    assert clf.is_trained

    # Test Medical Record text
    res_med = clf.predict("Patient Rx: Amoxicillin 500mg capsules twice daily. Diagnosis: bronchitis.")
    assert res_med.predicted_class == "Medical Record"
    assert res_med.confidence > 0.4

    # Test Invoice text
    res_inv = clf.predict("Tax Invoice # 984. Total amount due: $1,250.00. Payment due.")
    assert res_inv.predicted_class == "Invoice / Receipt"

"""Tests for Information Extraction module."""

import pytest
from src.extraction.extractor import InformationExtractor
from src.extraction.regex_extractor import RegexExtractor


def test_regex_extraction_patterns():
    extractor = RegexExtractor()
    sample = """
    Invoice Date: 2024-10-15
    Customer Contact: support@example.com, Phone: (555) 123-4567
    Total Amount: $1,450.50 USD
    Ref Code: INV-8941
    """
    matches = extractor.extract_all(sample)

    assert "2024-10-15" in matches["dates"]
    assert "support@example.com" in matches["emails"]
    assert any("555" in p for p in matches["phones"])
    assert any("1,450.50" in a for a in matches["amounts"])
    assert any("INV-8941" in r for r in matches["reference_codes"])


def test_key_value_extraction():
    extractor = InformationExtractor()
    sample = """
    Patient Name: John Doe
    Doctor Name: Jane Smith
    Age: 45
    Diagnosis: Acute Flu
    """
    fields = extractor.extract_all(sample)
    assert fields.key_value_pairs.get("Patient Name") == "John Doe"
    assert fields.key_value_pairs.get("Diagnosis") == "Acute Flu"

    df = fields.to_dataframe()
    assert not df.empty
    assert "Extracted Value" in df.columns

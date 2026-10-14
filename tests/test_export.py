"""Tests for ResultExporter (JSON, CSV, PDF)."""

import json
import pandas as pd
import pytest

from src.export.exporter import ResultExporter


def test_export_to_json_and_csv():
    data = {"doc": "test", "total": 100}
    json_out = ResultExporter.to_json(data)
    assert json.loads(json_out)["doc"] == "test"

    df = pd.DataFrame([{"Field": "Total", "Value": "$100"}])
    csv_out = ResultExporter.to_csv(df)
    assert "Total" in csv_out
    assert "$100" in csv_out


def test_export_to_pdf_report():
    df = pd.DataFrame([{"Field Category": "Financial", "Field Name": "Amount", "Extracted Value": "$150.00"}])
    pdf_bytes = ResultExporter.to_pdf_report(
        filename="invoice.png",
        document_class="Invoice / Receipt",
        confidence=0.89,
        full_text="Invoice text line",
        extracted_df=df,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500  # valid PDF generated
    assert pdf_bytes.startswith(b"%PDF")

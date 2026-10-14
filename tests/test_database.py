"""Tests for SQLite DatabaseManager and DocumentRecord."""

import pytest
from src.database.db_manager import DatabaseManager
from src.database.models import DocumentRecord


def test_sqlite_crud_operations(tmp_path):
    db_file = tmp_path / "test_docs.db"
    db = DatabaseManager(db_path=db_file)

    # 1. Insert record
    record = DocumentRecord(
        filename="sample_rx.png",
        page_count=1,
        full_text="Prescription for Amoxicillin 500mg",
        document_class="Medical Record",
        classification_confidence=0.92,
        extracted_fields={"dates": ["2024-10-15"], "amounts": ["$50.00"]},
    )
    rec_id = db.insert_document(record)
    assert rec_id > 0

    # 2. Retrieve all
    all_recs = db.get_all_documents()
    assert len(all_recs) == 1
    assert all_recs[0].filename == "sample_rx.png"
    assert all_recs[0].extracted_fields["dates"] == ["2024-10-15"]

    # 3. Retrieve by ID
    fetched = db.get_document_by_id(rec_id)
    assert fetched is not None
    assert fetched.document_class == "Medical Record"

    # 4. Search
    search_results = db.search_documents("Amoxicillin")
    assert len(search_results) == 1

    # 5. Delete
    deleted = db.delete_document(rec_id)
    assert deleted is True
    assert db.get_document_by_id(rec_id) is None

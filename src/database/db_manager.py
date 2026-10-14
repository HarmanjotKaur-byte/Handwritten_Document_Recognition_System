"""SQLite Database Manager for document record persistence.

Manages connection pooling, schema initialization, parameterized CRUD operations,
and text search queries.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

from src.config import DATABASE_PATH
from src.database.models import DocumentRecord


class DatabaseManager:
    """Handles all SQLite persistence operations with automatic schema creation."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        """Initialize the DatabaseManager.

        Args:
            db_path (Path): Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Establish connection with thread safety."""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_schema(self) -> None:
        """Initialize the documents table if it does not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 1,
                    full_text TEXT,
                    document_class TEXT,
                    classification_confidence REAL,
                    extracted_fields_json TEXT
                )
                """
            )
            # Create indices for fast lookup
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_class ON documents(document_class);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at);"
            )
            conn.commit()

    def insert_document(self, record: DocumentRecord) -> int:
        """Insert a newly processed document record.

        Args:
            record (DocumentRecord): Record to persist.

        Returns:
            int: Primary key ID of the inserted record.
        """
        query = """
            INSERT INTO documents (
                filename, created_at, page_count, full_text,
                document_class, classification_confidence, extracted_fields_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            record.filename,
            record.created_at,
            record.page_count,
            record.full_text,
            record.document_class,
            record.classification_confidence,
            record.get_extracted_fields_json(),
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            record_id = cursor.lastrowid
            record.id = record_id
            return int(record_id)

    def get_all_documents(self, limit: int = 100) -> List[DocumentRecord]:
        """Retrieve recent document records ordered by created_at descending.

        Args:
            limit (int): Maximum count of records.

        Returns:
            List[DocumentRecord]: Retrieved records.
        """
        query = """
            SELECT id, filename, created_at, page_count, full_text,
                   document_class, classification_confidence, extracted_fields_json
            FROM documents
            ORDER BY id DESC
            LIMIT ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [DocumentRecord.from_row(r) for r in rows]

    def get_document_by_id(self, doc_id: int) -> Optional[DocumentRecord]:
        """Fetch a specific document record by ID.

        Args:
            doc_id (int): Primary key ID.

        Returns:
            Optional[DocumentRecord]: Record if found, else None.
        """
        query = """
            SELECT id, filename, created_at, page_count, full_text,
                   document_class, classification_confidence, extracted_fields_json
            FROM documents
            WHERE id = ?
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (doc_id,))
            row = cursor.fetchone()
            if row:
                return DocumentRecord.from_row(row)
            return None

    def search_documents(self, query_str: str) -> List[DocumentRecord]:
        """Search documents by filename, class, or text contents.

        Args:
            query_str (str): Substring search phrase.

        Returns:
            List[DocumentRecord]: Matching records.
        """
        query = """
            SELECT id, filename, created_at, page_count, full_text,
                   document_class, classification_confidence, extracted_fields_json
            FROM documents
            WHERE filename LIKE ? OR document_class LIKE ? OR full_text LIKE ?
            ORDER BY id DESC
        """
        param = f"%{query_str}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (param, param, param))
            rows = cursor.fetchall()
            return [DocumentRecord.from_row(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document record by ID.

        Args:
            doc_id (int): Primary key ID.

        Returns:
            bool: True if deleted, False if not found.
        """
        query = "DELETE FROM documents WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

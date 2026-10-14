"""Data model representations for SQLite database records."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DocumentRecord:
    """Represents a persisted document processing entry in SQLite."""

    id: Optional[int] = None
    filename: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    page_count: int = 1
    full_text: str = ""
    document_class: str = "General / Unclassified"
    classification_confidence: float = 0.0
    extracted_fields: Dict[str, Any] = field(default_factory=dict)

    def get_extracted_fields_json(self) -> str:
        """Serialize extracted fields dictionary to JSON string."""
        return json.dumps(self.extracted_fields, ensure_ascii=False)

    @classmethod
    def from_row(cls, row: tuple) -> "DocumentRecord":
        """Construct instance from an SQLite query row tuple.

        Tuple schema:
        (id, filename, created_at, page_count, full_text, document_class,
         classification_confidence, extracted_fields_json)
        """
        try:
            fields_dict = json.loads(row[7]) if row[7] else {}
        except Exception:
            fields_dict = {}

        return cls(
            id=row[0],
            filename=row[1],
            created_at=row[2],
            page_count=row[3],
            full_text=row[4] or "",
            document_class=row[5] or "General / Unclassified",
            classification_confidence=float(row[6]) if row[6] is not None else 0.0,
            extracted_fields=fields_dict,
        )

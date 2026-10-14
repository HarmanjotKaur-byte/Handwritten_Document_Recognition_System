"""Rule-based pattern extraction for structured document fields using Regular Expressions.

Matches dates, phone numbers, email addresses, financial amounts/currencies,
and document reference codes.
"""

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExtractedEntity:
    """Represents a discrete extracted entity instance."""

    field_type: str
    value: str
    confidence: float = 1.0


class RegexExtractor:
    """Extracts structured values from text using compiled regular expressions."""

    def __init__(self):
        """Compile regex patterns for high-frequency document fields."""
        # Email address pattern
        self.email_regex = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        )

        # Phone numbers: various international and national formats
        self.phone_regex = re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        )

        # Dates: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, and 'Month DD, YYYY'
        self.date_regex = re.compile(
            r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
            r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
            r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        )

        # Currency & monetary values: $1,250.00, USD 450, 50.00 EUR
        self.amount_regex = re.compile(
            r"(?:\$|€|£|₹|USD|EUR|GBP|INR)\s?\d+(?:,\d{3})*(?:\.\d{2})?"
            r"|\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP|INR|dollars|cents)\b",
            re.IGNORECASE,
        )

        # Common Reference / ID / Invoice Codes (e.g. INV-1234, APP-9921, REF#881)
        self.ref_code_regex = re.compile(
            r"\b(?:INV|REF|APP|REC|RX|NO|ID|DOC)[-#:\s]+[A-Z0-9-]{3,15}\b",
            re.IGNORECASE,
        )

    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """Extract all matched regex entities from text.

        Args:
            text (str): Source text string.

        Returns:
            Dict[str, List[str]]: Entity groups mapping to lists of unique string values.
        """
        if not text:
            return {
                "emails": [],
                "phones": [],
                "dates": [],
                "amounts": [],
                "reference_codes": [],
            }

        # Deduplicate while preserving order
        def _unique(matches: List[str]) -> List[str]:
            seen = set()
            out = []
            for m in matches:
                m_str = m.strip()
                if m_str and m_str not in seen:
                    seen.add(m_str)
                    out.append(m_str)
            return out

        return {
            "emails": _unique(self.email_regex.findall(text)),
            "phones": _unique(self.phone_regex.findall(text)),
            "dates": _unique(self.date_regex.findall(text)),
            "amounts": _unique(self.amount_regex.findall(text)),
            "reference_codes": _unique(self.ref_code_regex.findall(text)),
        }

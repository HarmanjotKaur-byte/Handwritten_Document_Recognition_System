"""Unified Information Extraction pipeline.

Combines regular expression patterns, spaCy named entity recognition,
and key-value pair heuristics into structured Pandas DataFrames.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.extraction.ner_extractor import SpacyNERExtractor
from src.extraction.regex_extractor import RegexExtractor


@dataclass
class ExtractedFields:
    """Consolidated representation of all extracted entities and key-value fields."""

    dates: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    amounts: List[str] = field(default_factory=list)
    reference_codes: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    key_value_pairs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert structured fields to standard dictionary for serialization."""
        return {
            "dates": self.dates,
            "emails": self.emails,
            "phones": self.phones,
            "amounts": self.amounts,
            "reference_codes": self.reference_codes,
            "persons": self.persons,
            "organizations": self.organizations,
            "locations": self.locations,
            "key_value_pairs": self.key_value_pairs,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert extracted fields into a tabular DataFrame for Streamlit editing."""
        records = []
        for k, v in self.key_value_pairs.items():
            records.append({"Field Category": "Key-Value Pair", "Field Name": k, "Extracted Value": v})
        for v in self.persons:
            records.append({"Field Category": "Person Name", "Field Name": "Person", "Extracted Value": v})
        for v in self.organizations:
            records.append({"Field Category": "Organization", "Field Name": "Organization", "Extracted Value": v})
        for v in self.locations:
            records.append({"Field Category": "Location", "Field Name": "Location", "Extracted Value": v})
        for v in self.dates:
            records.append({"Field Category": "Date", "Field Name": "Date", "Extracted Value": v})
        for v in self.amounts:
            records.append({"Field Category": "Financial", "Field Name": "Amount", "Extracted Value": v})
        for v in self.emails:
            records.append({"Field Category": "Contact", "Field Name": "Email", "Extracted Value": v})
        for v in self.phones:
            records.append({"Field Category": "Contact", "Field Name": "Phone", "Extracted Value": v})
        for v in self.reference_codes:
            records.append({"Field Category": "Reference", "Field Name": "Reference Code", "Extracted Value": v})

        if not records:
            return pd.DataFrame(columns=["Field Category", "Field Name", "Extracted Value"])

        return pd.DataFrame(records)


class InformationExtractor:
    """High-level facade orchestrating Regex and NER extractors."""

    def __init__(self):
        """Initialize extractors."""
        self.regex_engine = RegexExtractor()
        self.ner_engine = SpacyNERExtractor()

        # Regex pattern for key-value labels: e.g., "Patient Name: John", "Total: $12"
        self.kv_pattern = re.compile(
            r"^(?P<key>[A-Za-z0-9\s/_-]{2,30})\s*[:=]\s*(?P<val>.+)$"
        )

    def extract_key_values(self, text: str) -> Dict[str, str]:
        """Parse lines conforming to Key: Value or Key = Value patterns.

        Args:
            text (str): Document text.

        Returns:
            Dict[str, str]: Extracted key-value mappings.
        """
        kv_pairs: Dict[str, str] = {}
        for line in text.splitlines():
            line_clean = line.strip()
            match = self.kv_pattern.match(line_clean)
            if match:
                k = match.group("key").strip()
                v = match.group("val").strip()
                if k and v:
                    kv_pairs[k] = v
        return kv_pairs

    def extract_all(self, text: str) -> ExtractedFields:
        """Run complete information extraction on text.

        Args:
            text (str): Extracted full text from handwriting recognition.

        Returns:
            ExtractedFields: Aggregated structured findings.
        """
        if not text:
            return ExtractedFields()

        regex_results = self.regex_engine.extract_all(text)
        ner_results = self.ner_engine.extract_entities(text)
        kv_pairs = self.extract_key_values(text)

        return ExtractedFields(
            dates=regex_results.get("dates", []),
            emails=regex_results.get("emails", []),
            phones=regex_results.get("phones", []),
            amounts=regex_results.get("amounts", []),
            reference_codes=regex_results.get("reference_codes", []),
            persons=ner_results.get("persons", []),
            organizations=ner_results.get("organizations", []),
            locations=ner_results.get("locations", []),
            key_value_pairs=kv_pairs,
        )

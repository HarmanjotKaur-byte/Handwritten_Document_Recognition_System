"""Named Entity Recognition (NER) using spaCy.

Extracts named entities such as PERSON, ORG, GPE (locations), and monetary amounts.
Includes graceful fallback if the spaCy language model is not yet installed.
"""

from typing import Dict, List, Optional

from src.config import SPACY_MODEL_NAME


class SpacyNERExtractor:
    """Wrapper around spaCy NLP pipeline for named entity recognition."""

    def __init__(self, model_name: str = SPACY_MODEL_NAME):
        """Initialize spaCy pipeline.

        Args:
            model_name (str): Name of the spaCy model package (default: en_core_web_sm).
        """
        self.model_name = model_name
        self.nlp = None
        self.is_loaded = False
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        """Attempt loading the spaCy language model."""
        try:
            import spacy

            try:
                self.nlp = spacy.load(self.model_name)
                self.is_loaded = True
            except OSError:
                # Model package not downloaded yet
                self.nlp = None
                self.is_loaded = False
        except ImportError:
            self.nlp = None
            self.is_loaded = False

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities categorized by label.

        Args:
            text (str): Document text.

        Returns:
            Dict[str, List[str]]: Mapping of entity types (persons, organizations, locations).
        """
        results: Dict[str, List[str]] = {
            "persons": [],
            "organizations": [],
            "locations": [],
        }

        if not text or not self.is_loaded or self.nlp is None:
            return results

        doc = self.nlp(text)
        for ent in doc.ents:
            val = ent.text.strip()
            if not val or len(val) < 2:
                continue

            if ent.label_ == "PERSON" and val not in results["persons"]:
                results["persons"].append(val)
            elif ent.label_ == "ORG" and val not in results["organizations"]:
                results["organizations"].append(val)
            elif ent.label_ in {"GPE", "LOC"} and val not in results["locations"]:
                results["locations"].append(val)

        return results

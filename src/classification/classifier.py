"""Document classification using scikit-learn TF-IDF and Logistic Regression.

Classifies recognized text into categories (Medical Record, Invoice, Application Form,
Handwritten Note, etc.) with confidence threshold validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    CLASSIFIER_MODEL_PATH,
    DOCUMENT_CLASSES,
    VECTORIZER_MODEL_PATH,
)


@dataclass
class ClassificationResult:
    """Outcome of document classification."""

    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    is_trained: bool = False
    status_message: str = ""


class DocumentClassifier:
    """Manages training artifacts and inference for document category classification."""

    def __init__(
        self,
        classifier_path: Optional[Path] = CLASSIFIER_MODEL_PATH,
        vectorizer_path: Optional[Path] = VECTORIZER_MODEL_PATH,
        confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
    ):
        """Initialize the document classifier.

        Args:
            classifier_path (Optional[Path]): Path to saved LogisticRegression joblib file.
            vectorizer_path (Optional[Path]): Path to saved TfidfVectorizer joblib file.
            confidence_threshold (float): Minimum confidence to assign a specific class.
        """
        self.classifier_path = Path(classifier_path) if classifier_path else None
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path else None
        self.confidence_threshold = confidence_threshold

        self.classifier: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_trained: bool = False

        self._load_models_if_available()

    def _load_models_if_available(self) -> None:
        """Load fitted models if joblib files exist on disk."""
        if (
            self.classifier_path
            and self.classifier_path.is_file()
            and self.vectorizer_path
            and self.vectorizer_path.is_file()
        ):
            try:
                self.classifier = joblib.load(self.classifier_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_trained = True
            except Exception:
                self.classifier = None
                self.vectorizer = None
                self.is_trained = False
        else:
            self.classifier = None
            self.vectorizer = None
            self.is_trained = False

    def predict(self, text: str) -> ClassificationResult:
        """Classify input text string.

        Args:
            text (str): Extracted document text.

        Returns:
            ClassificationResult: Predicted category, confidence, and probability distribution.
        """
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return ClassificationResult(
                predicted_class="General / Unclassified",
                confidence=0.0,
                class_probabilities={},
                is_trained=self.is_trained,
                status_message="Empty text provided. Cannot classify.",
            )

        if not self.is_trained or self.classifier is None or self.vectorizer is None:
            return ClassificationResult(
                predicted_class="General / Unclassified",
                confidence=0.0,
                class_probabilities={},
                is_trained=False,
                status_message="Classifier is not trained. Train the model using train_classifier.py.",
            )

        # Vectorize text using fitted TF-IDF
        features = self.vectorizer.transform([cleaned_text])
        probabilities = self.classifier.predict_proba(features)[0]
        classes = self.classifier.classes_

        prob_dict = {str(c): float(p) for c, p in zip(classes, probabilities)}

        best_idx = probabilities.argmax()
        best_class = str(classes[best_idx])
        best_prob = float(probabilities[best_idx])

        # Apply confidence threshold fallback
        if best_prob < self.confidence_threshold:
            assigned_class = "General / Unclassified"
            status = f"Low confidence ({best_prob:.2f} < {self.confidence_threshold:.2f}). Fallback assigned."
        else:
            assigned_class = best_class
            status = f"Classified with {best_prob:.2%} confidence."

        return ClassificationResult(
            predicted_class=assigned_class,
            confidence=best_prob,
            class_probabilities=prob_dict,
            is_trained=True,
            status_message=status,
        )

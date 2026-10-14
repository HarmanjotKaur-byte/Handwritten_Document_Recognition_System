"""Verify that HandwritingRecognizer loads CodeMasterCRNN correctly."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr.recognizer import HandwritingRecognizer

rec = HandwritingRecognizer()
diag = rec.get_status_diagnostics()
print("HandwritingRecognizer Diagnostics:")
for k, v in diag.items():
    print(f"  {k}: {v}")

print(f"Model loaded: {rec.model_loaded}")
print(f"Is trained: {rec.is_trained}")
if rec.iam_predictor:
    print(f"IAM Predictor status: {rec.iam_predictor.status_message}")
    print(f"Is CodeMaster: {rec.iam_predictor.is_codemaster}")

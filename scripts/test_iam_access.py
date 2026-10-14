"""Inspect access, splits, schema, and sample data for the Teklia/IAM-line dataset."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset


def inspect_iam_dataset():
    print("=" * 60)
    print("Inspecting Hugging Face Dataset: Teklia/IAM-line")
    print("=" * 60)

    try:
        print("\n[1] Loading dataset...")
        dataset = load_dataset("Teklia/IAM-line")
        print("    Successfully loaded dataset from Hugging Face!")
    except Exception as e:
        print(f"\n[!] Failed to load dataset: {e}")
        sys.exit(1)

    print("\n[2] Available Splits:")
    for split_name in dataset.keys():
        print(f"    - {split_name}: {len(dataset[split_name])} examples")

    # Inspect schema on train split (or first available split)
    first_split = list(dataset.keys())[0]
    sample = dataset[first_split][0]

    print(f"\n[3] Schema & Column Names ({first_split} split):")
    for col_name, value in sample.items():
        val_type = type(value).__name__
        if hasattr(value, "size"):
            val_type += f" (size: {value.size}, mode: {getattr(value, 'mode', 'N/A')})"
        print(f"    - Column '{col_name}': {val_type}")

    # Identify image and transcription columns dynamically
    img_col = None
    text_col = None

    for col in sample.keys():
        if "image" in col.lower() or hasattr(sample[col], "size"):
            img_col = col
        if any(term in col.lower() for term in ["text", "transcription", "label", "ground_truth"]):
            text_col = col

    print(f"\n[4] Identified Columns:")
    print(f"    - Image Column: '{img_col}'")
    print(f"    - Transcription Column: '{text_col}'")

    print(f"\n[5] Sample Inspections from '{first_split}':")
    for i in range(min(5, len(dataset[first_split]))):
        ex = dataset[first_split][i]
        img = ex[img_col]
        txt = ex[text_col]
        print(f"    Sample #{i + 1}:")
        print(f"      Transcription: '{txt}'")
        print(f"      Image Mode: {img.mode}, Size: {img.size} (W x H)")

    # Data validation checks over first 100 samples
    print(f"\n[6] Validation check over first 100 samples of '{first_split}':")
    missing_labels = 0
    zero_size_images = 0
    char_set = set()

    check_limit = min(100, len(dataset[first_split]))
    for i in range(check_limit):
        ex = dataset[first_split][i]
        txt = ex[text_col]
        img = ex[img_col]
        if txt is None or len(str(txt).strip()) == 0:
            missing_labels += 1
        else:
            char_set.update(list(str(txt)))
        if img.size[0] == 0 or img.size[1] == 0:
            zero_size_images += 1

    print(f"    - Missing / empty labels: {missing_labels}")
    print(f"    - Invalid 0-size images: {zero_size_images}")
    print(f"    - Unique characters found in check: {len(char_set)}")
    print(f"    - Characters sample: {''.join(sorted(list(char_set)))}")

    print("\n[7] Dataset verification completed successfully.")


if __name__ == "__main__":
    inspect_iam_dataset()

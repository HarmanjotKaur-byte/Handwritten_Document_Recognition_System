"""Generate the Google Colab Notebook for the TrOCR Base Handwritten POC."""

import json
import os

with open('experiments/trocr_poc/embedded_samples.json', 'r', encoding='utf-8') as f:
    embedded = json.load(f)

nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'gpuType': 'T4', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}
    },
    'cells': []
}


def add_md(text):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [text]})


def add_code(code):
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [code]})


add_md(
    '# Proof-of-Concept: Official Pretrained `microsoft/trocr-base-handwritten`\n\n'
    '**Project:** AI-Based Handwritten Document Recognition System\n\n'
    '### Objective:\n'
    'Isolate and test Microsoft\'s official Vision Transformer (`microsoft/trocr-base-handwritten`) on GPU.\n\n'
    '### Test Suite:\n'
    '1. **2 IAM Benchmark Lines** (standard English handwriting dataset)\n'
    '2. **2 User Real Handwritten Lines** (notebook paper with ballpoint/gel pen)\n'
    '3. **1 Machine-Printed Baseline Line** (testing handwriting model on printed text)\n\n'
    '> **Rule:** Zero fine-tuning. Pure out-of-the-box Hugging Face inference pipeline.'
)

add_md('## Step 1: Check GPU Runtime')
add_code(
    'import torch\n'
    'print("PyTorch Version:", torch.__version__)\n'
    'print("CUDA Available:", torch.cuda.is_available())\n'
    'if torch.cuda.is_available():\n'
    '    print("GPU Device:", torch.cuda.get_device_name(0))\n'
    'else:\n'
    '    print("WARNING: Running on CPU! Enable GPU: Runtime -> Change runtime type -> T4 GPU")\n'
)

add_md('## Step 2: Install Hugging Face Transformers & Dependencies')
add_code('!pip install -q transformers torchvision pillow jiwer')

add_md(
    '## Step 3: Unpack Embedded Test Line Images\n'
    '*(Unpacks 2 IAM lines, 2 user handwriting lines, and 1 printed text line automatically)*'
)

unpack_code = (
    'import os, base64\n\n'
    'os.makedirs("test_images", exist_ok=True)\n\n'
    f'samples_data = {json.dumps(embedded)}\n\n'
    'for fname, info in samples_data.items():\n'
    '    raw_bytes = base64.b64decode(info["b64"])\n'
    '    out_path = os.path.join("test_images", fname)\n'
    '    with open(out_path, "wb") as f:\n'
    '        f.write(raw_bytes)\n'
    '    print(f"Extracted: {fname} [{info[\'type\']}] -> Ground Truth: \'{info[\'gt\']}\'")\n'
)
add_code(unpack_code)

add_md('## Step 4: Visual Preview of Test Images')
add_code(
    'import matplotlib.pyplot as plt\n'
    'from PIL import Image\n\n'
    'fig, axes = plt.subplots(len(samples_data), 1, figsize=(12, 10))\n'
    'for ax, (fname, info) in zip(axes, samples_data.items()):\n'
    '    img_p = os.path.join("test_images", fname)\n'
    '    im = Image.open(img_p)\n'
    '    ax.imshow(im)\n'
    '    ax.set_title(f"[{info[\'type\']}] {fname} | Dimensions: {im.size[0]}x{im.size[1]} | GT: \'{info[\'gt\']}\'", fontsize=11, loc="left")\n'
    '    ax.axis("off")\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
)

add_md(
    '## Step 5: Load Official Pretrained TrOCR Model & Processor\n'
    '*(Downloads `microsoft/trocr-base-handwritten` from Hugging Face)*'
)
add_code(
    'import time\n'
    'from transformers import TrOCRProcessor, VisionEncoderDecoderModel\n\n'
    'print("Loading official TrOCRProcessor & VisionEncoderDecoderModel from Hugging Face...")\n'
    'start_load = time.time()\n\n'
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    'processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")\n'
    'model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)\n'
    'model.eval()\n\n'
    'load_time = time.time() - start_load\n'
    'print(f"Model mounted on {device} in {load_time:.2f} seconds!")\n'
    'print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")\n'
)

add_md(
    '## Step 6: Run Official Inference Pipeline\n'
    'We strictly adhere to the official Microsoft TrOCR pipeline:\n'
    '1. Convert image to RGB PIL Image (no manual custom resizing/binarization).\n'
    '2. Process through `processor(images=image, return_tensors="pt").pixel_values`.\n'
    '3. Generate token sequences via `model.generate()`.\n'
    '4. Decode with `processor.batch_decode(skip_special_tokens=True)`.'
)

add_code(
    'import time, json\n'
    'import jiwer\n\n'
    'results = []\n\n'
    'print("=" * 90)\n'
    'print(f"{\'IMAGE\':<20} | {\'DIMS\':<10} | {\'PREDICTED TEXT\':<32} | {\'LATENCY\':<8}")\n'
    'print("=" * 90)\n\n'
    'for fname, info in samples_data.items():\n'
    '    img_path = os.path.join("test_images", fname)\n'
    '    gt_text = info["gt"]\n'
    '    sample_type = info["type"]\n\n'
    '    img = Image.open(img_path).convert("RGB")\n'
    '    width, height = img.size\n\n'
    '    error_msg = None\n'
    '    pred_text = ""\n'
    '    latency_ms = 0.0\n\n'
    '    try:\n'
    '        t0 = time.time()\n'
    '        # Official TrOCR Preprocessing\n'
    '        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)\n\n'
    '        # Generation\n'
    '        with torch.no_grad():\n'
    '            generated_ids = model.generate(pixel_values, max_new_tokens=64)\n\n'
    '        # Token Decoding\n'
    '        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()\n'
    '        latency_ms = (time.time() - t0) * 1000.0\n'
    '    except Exception as e:\n'
    '        error_msg = str(e)\n\n'
    '    cer = jiwer.cer(gt_text, pred_text) if not error_msg else 1.0\n'
    '    wer = jiwer.wer(gt_text, pred_text) if not error_msg else 1.0\n\n'
    '    record = {\n'
    '        "filename": fname,\n'
    '        "sample_type": sample_type,\n'
    '        "dimensions": f"{width}x{height}",\n'
    '        "ground_truth": gt_text,\n'
    '        "predicted_text": pred_text,\n'
    '        "cer": round(cer, 4),\n'
    '        "wer": round(wer, 4),\n'
    '        "latency_ms": round(latency_ms, 1),\n'
    '        "error": error_msg\n'
    '    }\n'
    '    results.append(record)\n'
    '    print(f"{fname:<20} | {f\'{width}x{height}\':<10} | {pred_text:<32} | {f\'{latency_ms:.1f}ms\':<8}")\n\n'
    'print("=" * 90)\n'
)

add_md('## Step 7: Results Table & Detailed Inspection')
add_code(
    'import pandas as pd\n'
    'from IPython.display import display\n\n'
    'df = pd.DataFrame(results)[["filename", "sample_type", "dimensions", "ground_truth", "predicted_text", "cer", "wer", "latency_ms"]]\n'
    'df["cer_pct"] = (df["cer"] * 100).round(1).astype(str) + "%"\n'
    'df["wer_pct"] = (df["wer"] * 100).round(1).astype(str) + "%"\n\n'
    'display(df[["filename", "sample_type", "dimensions", "ground_truth", "predicted_text", "cer_pct", "wer_pct", "latency_ms"]])\n\n'
    'with open("trocr_poc_results.json", "w") as f:\n'
    '    json.dump(results, f, indent=2)\n\n'
    'print("\\nSaved detailed results to \'trocr_poc_results.json\'.")\n'
)

add_md('## Step 8: Download Predictions & Summary')
add_code(
    'from google.colab import files\n'
    'files.download("trocr_poc_results.json")\n'
)

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/TrOCR_Base_Handwritten_POC.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Generated notebooks/TrOCR_Base_Handwritten_POC.ipynb successfully!')

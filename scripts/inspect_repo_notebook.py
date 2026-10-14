"""Inspect notebook cells, model definition, checkpoint, and outputs in CodeMaster-AJ repo."""

import json
import os
import torch
import torch.nn as nn

nb_path = "external/handwritten-text-recognition/main_project.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    ctype = cell["cell_type"]
    source = "".join(cell.get("source", []))
    lines = [l for l in source.split("\n") if l.strip()]
    first_line = lines[0] if lines else "EMPTY"
    out_types = [o.get("output_type", "") for o in cell.get("outputs", [])]
    print(f"\n--- CELL {idx} ({ctype}) ---")
    print(f"First line: {first_line}")
    if out_types:
        print(f"Outputs ({len(out_types)}): {out_types}")
        for o in cell.get("outputs", []):
            if "text" in o:
                text_content = "".join(o["text"]).strip()
                print("Text Output Sample:\n", text_content[:400])

print("\n================ CHECKPOINT INSPECTION ================")
ckpt_path = "external/handwritten-text-recognition/saved_models/best_model.pth"
print("File exists:", os.path.exists(ckpt_path))
if os.path.exists(ckpt_path):
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB ({os.path.getsize(ckpt_path)} bytes)")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print(f"Number of keys: {len(ckpt.keys())}")
        total_params = 0
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:30s} -> shape: {str(list(v.shape)):18s} dtype: {v.dtype}")
                total_params += v.numel()
        print(f"Total parameters in checkpoint: {total_params:,}")

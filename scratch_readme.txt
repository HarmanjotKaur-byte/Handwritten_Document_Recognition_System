---
language:
- en
license: mit
tags:
- handwriting-recognition
- ocr
- computer-vision
- pytorch
- crnn
- ctc
- iam-dataset
library_name: pytorch
datasets:
- Teklia/IAM-line
metrics:
- cer
- wer
model-index:
- name: handwriting-recognition-iam
  results:
  - task:
      type: image-to-text
      name: Handwriting Recognition
    dataset:
      name: IAM Handwriting Database
      type: Teklia/IAM-line
    metrics:
    - type: cer
      value: 0.1295
      name: Character Error Rate
    - type: wer
      value: 0.4247
      name: Word Error Rate
---

# Handwriting Recognition (CNN-BiLSTM-CTC)

A complete handwriting recognition system using a CNN-BiLSTM-CTC architecture trained on the IAM handwriting database. Achieves **12.95% CER** and **42.47% WER** on the IAM test set.

**Hugging Face**: [IsmatS/handwriting_recognition](https://huggingface.co/IsmatS/handwriting_recognition)

## Quick Start

```bash
pip install torch torchvision pillow numpy huggingface_hub
```

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Download model checkpoint
ckpt_path = hf_hub_download(repo_id="IsmatS/handwriting-recognition-iam", filename="best_model.pth")
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Character mapper from checkpoint
char_mapper = checkpoint['char_mapper']

def preprocess_image(image_path, target_height=64, target_width=256):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((target_width, target_height), Image.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize to [-1, 1]
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

def ctc_decode(predictions, char_mapper):
    """Greedy CTC decoding."""
    pred_indices = predictions.argmax(dim=2).squeeze(1).tolist()
    decoded = []
    prev = None
    for idx in pred_indices:
        if idx != prev and idx != 0:  # 0 = blank token
            decoded.append(char_mapper.idx_to_char[idx])
        prev = idx
    return ''.join(decoded)

# Note: CRNN class must match the training definition in train_colab.ipynb
# See train_colab.ipynb for the full model class
# model = CRNN(num_chars=len(char_mapper.chars))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
#
# img_tensor = preprocess_image("handwriting_sample.png")
# with torch.no_grad():
#     output = model(img_tensor)  # (T, 1, num_chars)
# text = ctc_decode(output, char_mapper)
# print("Recognized:", text)
```

> **Note**: The trained model weights (`best_model.pth`) are generated during training. Run `train_colab.ipynb` on Google Colab to produce the checkpoint, then use the code above for inference.

## 📁 Files

### 1. **analysis.ipynb** - Dataset Analysis
- Exploratory Data Analysis (EDA)
- 5 detailed charts saved to `charts/` folder
- Run locally or on Colab (no GPU needed)

### 2. **train_colab.ipynb** - Model Training (GPU)
- **⚡ Google Colab GPU compatible**
- Full training pipeline
- CNN-BiLSTM-CTC model (~9.1M parameters)
- Automatic model saving
- Download trained model for deployment

## 🚀 Quick Start

### Option 1: Analyze Dataset (Local/Colab)
```bash
jupyter notebook analysis.ipynb
```
- No GPU needed
- Generates 5 EDA charts
- Fast (~2 minutes)

### Option 2: Train Model (Google Colab GPU)

1. **Upload `train_colab.ipynb` to Google Colab**
2. **Change runtime to GPU:**
   - Runtime → Change runtime type → GPU (T4 recommended)
3. **Run all cells**
4. **Download trained model** (last cell)

**Training Time:** ~1-2 hours for 20 epochs on T4 GPU

## 📊 Charts Generated

From `analysis.ipynb`:
1. `charts/01_sample_images.png` - 10 sample handwritten texts
2. `charts/02_text_length_distribution.png` - Text statistics
3. `charts/03_image_dimensions.png` - Image analysis
4. `charts/04_character_frequency.png` - Character distribution
5. `charts/05_summary_statistics.png` - Summary table

## 🎯 Model Details

**Architecture:**
- **CNN**: 7 convolutional blocks (feature extraction)
- **BiLSTM**: 2 layers, 256 hidden units (sequence modeling)
- **CTC Loss**: Alignment-free training

**Dataset:** Teklia/IAM-line (Hugging Face)
- Train: 6,482 samples
- Validation: 976 samples
- Test: 2,915 samples

**Metrics:**
- **CER** (Character Error Rate)
- **WER** (Word Error Rate)

## 💾 Model Files

After training in Colab:
- `best_model.pth` - Trained model weights
- `training_history.png` - Loss/CER/WER plots
- `predictions.png` - Sample predictions

## 📦 Requirements

```
torch>=2.0.0
datasets>=2.14.0
pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
jupyter>=1.0.0
jiwer>=3.0.0
```

## 🔧 Usage

### Load Trained Model
```python
import torch

# Load checkpoint
checkpoint = torch.load('best_model.pth')
char_mapper = checkpoint['char_mapper']

# Create model
from train_colab import CRNN  # Copy model class
model = CRNN(num_chars=len(char_mapper.chars))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
# ... (preprocessing + inference)
```

## 📝 Notes

- **GPU strongly recommended** for training (use Colab T4)
- Training on CPU will be extremely slow (~20x slower)
- Colab free tier: 12-hour limit, sufficient for 20 epochs
- Model checkpoint includes character mapper for deployment

## 🎓 Training Tips

1. **Start with fewer epochs** (5-10) to test
2. **Monitor CER/WER** - stop if not improving
3. **Increase epochs** if still improving (up to 50)
4. **Save checkpoint** before Colab disconnects
5. **Download model immediately** after training

## 📄 License

Dataset: IAM Database (research use)

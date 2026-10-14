# AI-Powered Handwritten Document Understanding System

An intelligent, modular system for processing handwritten document images and multi-page PDFs, recognizing text, classifying document types, extracting structured metadata, editing results, and persisting/exporting data.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Web interface, interactive visualization, and editable data tables |
| **Ingestion & Preprocessing** | [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/), [NumPy](https://numpy.org/), [PyMuPDF](https://pymupdf.readthedocs.io/) | Document deskewing, denoising, adaptive binarization, line segmentation, and PDF rasterization |
| **Handwriting OCR** | [PyTorch](https://pytorch.org/) (CNN + BiLSTM + CTC) | Line-level handwritten character sequence recognition using Connectionist Temporal Classification |
| **Document Classification** | [scikit-learn](https://scikit-learn.org/) (TF-IDF + Logistic Regression) | Text vectorization and multi-class category prediction with confidence scoring |
| **Information Extraction** | Regular Expressions (`re`) & [spaCy](https://spacy.io/) | Structured field extraction (dates, amounts, contacts) and named entity recognition |
| **Data Presentation** | [Pandas](https://pandas.pydata.org/) | In-memory tabular structuring and data manipulation |
| **Persistence & Storage** | [SQLite](https://www.sqlite.org/) | Local relational database for document records and search |
| **Export Formats** | [ReportLab](https://www.reportlab.com/) & JSON/CSV | Exportable summary reports and structured data payloads |

---

## 📂 Project Structure

```text
Handwritten_Document_Understanding_System/
├── app.py                         # Streamlit web application entry point
├── requirements.txt               # Pinned project dependencies
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for virtualenvs, caches, and weights
├── README.md                      # Project documentation and guide
│
├── data/
│   ├── database/                  # SQLite storage directory
│   │   └── documents.db           # Persistent SQLite database file
│   └── sample_documents/          # Directory for sample images and PDFs
│
├── models/                        # Serialized model checkpoints & weights
│   ├── crnn_handwriting.pth       # PyTorch CRNN trained weights
│   ├── tfidf_vectorizer.joblib    # scikit-learn TF-IDF vectorizer
│   └── document_classifier.joblib # scikit-learn Logistic Regression classifier
│
├── src/                           # Core application logic packages
│   ├── config.py                  # Global configurations, paths, device detection, constants
│   ├── io/                        # Document and image ingestion
│   │   ├── __init__.py
│   │   └── loader.py              # Ingests images and multi-page PDFs via PyMuPDF
│   ├── preprocessing/             # Image processing and layout segmentation
│   │   ├── __init__.py
│   │   ├── image_processor.py     # Grayscale, denoising, deskewing, binarization, normalization
│   │   └── segmenter.py           # Horizontal projection profiling & line crop segmentation
│   ├── ocr/                       # PyTorch handwriting recognition (CRNN + CTC)
│   │   ├── __init__.py
│   │   ├── model.py               # CRNN neural architecture (7-layer CNN + 2-layer BiLSTM + Linear)
│   │   ├── decoder.py             # Greedy CTC sequence decoder and character mapping
│   │   ├── dataset.py             # PyTorch Dataset and dynamic batch collator
│   │   ├── recognizer.py          # High-level inference engine with model diagnostic reporting
│   │   └── train_crnn.py          # Standalone training script with synthetic data generator
│   ├── classification/            # Document type categorization
│   │   ├── __init__.py
│   │   ├── classifier.py          # TF-IDF + Logistic Regression inference with confidence fallback
│   │   └── train_classifier.py    # Training script on domain document corpus
│   ├── extraction/                # Structured information extraction
│   │   ├── __init__.py
│   │   ├── regex_extractor.py     # Regex patterns for dates, phones, emails, amounts, ref codes
│   │   ├── ner_extractor.py       # spaCy NER integration (names, organizations, locations)
│   │   └── extractor.py           # Unified extractor & key-value pair parser producing DataFrames
│   ├── database/                  # SQLite database persistence
│   │   ├── __init__.py
│   │   ├── models.py              # DocumentRecord dataclass and JSON serialization
│   │   └── db_manager.py          # Parameterized SQLite CRUD operations and search
│   ├── export/                    # Multi-format report downloads
│   │   ├── __init__.py
│   │   └── exporter.py            # JSON, CSV, and ReportLab PDF report generation
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       └── helpers.py             # Image array conversions, file size, and timestamp helpers
│
└── tests/                         # Automated unit & integration tests
    ├── __init__.py
    ├── test_loader.py             # Tests for DocumentLoader
    ├── test_preprocessing.py      # Tests for image enhancement & line segmentation
    ├── test_ocr.py                # Tests for CRNN architecture tensor shapes & CTC decoder
    ├── test_classification.py     # Tests for TF-IDF + Logistic Regression
    ├── test_extraction.py         # Tests for regex patterns & key-value parsing
    ├── test_database.py           # Tests for SQLite CRUD & search operations
    └── test_export.py             # Tests for JSON, CSV, and PDF report creation
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Virtual Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download spaCy Language Model
```powershell
python -m spacy download en_core_web_sm
```

### 5. Initialize Pre-trained Models
Train the baseline Document Classifier:
```powershell
python src/classification/train_classifier.py
```

Generate baseline CRNN checkpoint (optional, for pipeline validation):
```powershell
python src/ocr/train_crnn.py --epochs 15
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 🧪 Running the Automated Test Suite

Run the full suite of unit and integration tests:
```powershell
pytest -v
```

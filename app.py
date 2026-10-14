"""Streamlit Application for AI-Powered Handwritten Document Understanding.

Integrates:
- Document Ingestion & Validation (PNG, JPG, JPEG, and multi-page PDFs)
- Handwriting-Preserving Preprocessing (Grayscale, Bilateral Denoising, CLAHE, Safe Deskewing)
- Text Line Segmentation & Bounding Box Overlays
- PyTorch CRNN Handwriting Recognition (Trained on Teklia/IAM-line)
- Document Classification (TF-IDF + Logistic Regression)
- Structured Information Extraction (Regex + spaCy NER + Key-Value Pairing)
- SQLite Database Record Persistence & History Browser
- Structured Data Export (JSON, CSV, and ReportLab PDF Reports)
"""

import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.classification.classifier import DocumentClassifier
from src.config import (
    CLASSIFIER_MODEL_PATH,
    DATABASE_PATH,
    OCR_MODEL_PATH,
    get_torch_device,
)
from src.database.db_manager import DatabaseManager
from src.database.models import DocumentRecord
from src.export.exporter import ResultExporter
from src.extraction.extractor import InformationExtractor
from src.io.loader import (
    ALLOWED_EXTENSIONS,
    DocumentLoader,
    DocumentValidationError,
    LoadedDocument,
)
from src.ocr.recognizer import HandwritingRecognizer
from src.preprocessing.image_processor import (
    ImageProcessor,
    PreprocessedDocument,
    PreprocessingOptions,
)
from src.preprocessing.segmenter import LineSegmenter
from src.recognition.predictor import get_predictor
from src.utils.helpers import format_file_size, format_timestamp, numpy_to_pil

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Handwritten Document Understanding",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all_engines():
    """Initialize and cache all backend system engines."""
    loader = DocumentLoader(target_dpi=150, max_file_size_bytes=25 * 1024 * 1024)
    processor = ImageProcessor()
    segmenter = LineSegmenter()
    recognizer = HandwritingRecognizer(model_path=OCR_MODEL_PATH)
    classifier = DocumentClassifier(classifier_path=CLASSIFIER_MODEL_PATH)
    extractor = InformationExtractor()
    db = DatabaseManager(db_path=DATABASE_PATH)
    iam_predictor = get_predictor("models/checkpoints/pylaia_iam_crnn.pt")
    return loader, processor, segmenter, recognizer, classifier, extractor, db, iam_predictor


loader, processor, segmenter, recognizer, classifier, extractor, db, iam_predictor = load_all_engines()

# ---------------------------------------------------------
# Sidebar: Controls & Diagnostics
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📥 Ingestion & Controls")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Document (Image or PDF)",
        type=[ext.lstrip(".") for ext in sorted(ALLOWED_EXTENSIONS)],
        help="Allowed formats: PNG, JPG, JPEG, and PDF. Maximum size: 25 MB.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing Options")

    # 0. Illumination & Shadow Flattening
    apply_flatten_illumination = st.checkbox(
        "Flatten Shadows & Lighting",
        value=True,
        help="Normalizes camera shadows, dark corners, and uneven lighting across the document page.",
    )

    # 1. Contrast Enhancement (CLAHE)
    apply_clahe = st.checkbox("Contrast Enhancement (CLAHE)", value=True)
    clahe_clip = 2.0
    if apply_clahe:
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0, 0.2)

    # 2. Denoising
    apply_denoise = st.checkbox("Noise Reduction", value=True)
    denoise_method = "bilateral"
    if apply_denoise:
        denoise_method = st.selectbox(
            "Filter Type",
            options=["bilateral", "gaussian"],
            format_func=lambda x: "Bilateral (Edge-Preserving)" if x == "bilateral" else "Gaussian (3x3)",
        )

    # 3. Deskewing
    apply_deskew = st.checkbox("Safe Auto-Deskew", value=True)

    # 4. Optional Binarization
    binarize_mode = st.selectbox(
        "Binarization Method",
        options=["none", "otsu", "adaptive", "sauvola"],
        index=0,
        format_func=lambda x: {
            "none": "None (Enhanced Grayscale)",
            "otsu": "Otsu (Global Threshold)",
            "adaptive": "Adaptive Gaussian",
            "sauvola": "Sauvola (Degraded)",
        }.get(x, x),
    )

    max_dimension = st.slider("Max Dimension (px)", 1000, 2500, 1800, 100)

    st.markdown("---")
    st.markdown("### 🤖 OCR Recognition Engine")
    engine_choice = st.selectbox(
        "Select Model Architecture",
        options=["trocr", "codemaster", "pylaia"],
        index=0,
        format_func=lambda x: {
            "trocr": "🌟 TrOCR (Vision Transformer - Recommended SOTA)",
            "codemaster": "⚡ CodeMaster-CRNN (IAM Word Model)",
            "pylaia": "⚡ PyLaia-CRNN (IAM Benchmark Lines)",
        }.get(x, x),
    )
    apply_lexicon_opt = st.checkbox(
        "English Lexicon / Spell Correction",
        value=True,
        help="Applies dictionary and frequency priors to correct character ambiguities ('posiire' -> 'positive').",
    )

    st.markdown("---")
    st.markdown("### System Diagnostics")
    st.caption(f"**Compute Device:** `{get_torch_device()}`")

    if iam_predictor.is_ready:
        st.success("CRNN OCR: IAM-line checkpoint active")
    elif recognizer.is_trained:
        st.success("CRNN OCR: Baseline weights active")
    else:
        st.warning("CRNN OCR: Untrained")

    if classifier.is_trained:
        st.success("Classifier: Trained model loaded")
    else:
        st.warning("Classifier: Untrained")


# ---------------------------------------------------------
# Main UI Navigation Tabs
# ---------------------------------------------------------
tab_process, tab_history, tab_diagnostics = st.tabs(
    ["📄 Process Document", "🗄️ Document Records (SQLite)", "⚙️ Model Training & Diagnostics"]
)

# ---------------------------------------------------------
# TAB 1: Document Processing Pipeline
# ---------------------------------------------------------
with tab_process:
    st.markdown('<div class="main-title">AI Handwritten Document Understanding</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">End-to-end preprocessing, line segmentation, CRNN recognition, classification, and extraction.</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.info("👆 Please upload a handwritten document image or PDF from the sidebar to begin.")
    else:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        try:
            loaded_doc: LoadedDocument = loader.load_from_bytes(file_bytes, filename=file_name)
        except DocumentValidationError as ve:
            st.error(f"Validation Error: {ve}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading document: {e}")
            st.stop()

        # Multi-page PDF selector
        active_page_idx = 0
        if loaded_doc.is_pdf and loaded_doc.page_count > 1:
            col_p1, col_p2 = st.columns([1, 3])
            with col_p1:
                active_page_idx = st.selectbox(
                    "Select PDF Page",
                    options=list(range(loaded_doc.page_count)),
                    format_func=lambda i: f"Page {i + 1} of {loaded_doc.page_count}",
                )
            with col_p2:
                st.caption(f"Document contains {loaded_doc.page_count} pages. Currently viewing page {active_page_idx + 1}.")
            active_image = loaded_doc.pages[active_page_idx]
        else:
            active_image = loaded_doc.pages[0]

        # Stage 1: Preprocessing & Line Segmentation
        prep_options = PreprocessingOptions(
            apply_flatten_illumination=apply_flatten_illumination,
            apply_grayscale=True,
            apply_denoise=apply_denoise,
            denoise_method=denoise_method,
            apply_contrast_enhancement=apply_clahe,
            clahe_clip_limit=clahe_clip,
            apply_deskew=apply_deskew,
            binarize_method=binarize_mode,
            max_dimension=max_dimension,
        )

        with st.spinner("Applying handwriting-safe preprocessing & line segmentation..."):
            prep_doc = processor.process_page(active_image, options=prep_options)
            # Use deskewed document for accurate horizontal line valleys
            deskewed_source = prep_doc.deskewed if prep_doc.deskewed is not None else prep_doc.processed
            bin_for_seg, _ = processor.binarize(deskewed_source, method="otsu")
            segments = segmenter.segment_lines(bin_for_seg, deskewed_source)

        # Stage 2: Handwriting Recognition (TrOCR Vision Transformer)
        recognizer.engine_type = engine_choice
        diag = recognizer.get_status_diagnostics()
        with st.spinner(f"Running {diag['engine']} inference over {len(segments)} line strips..."):
            ocr_result = recognizer.recognize_lines(segments, apply_lexicon=apply_lexicon_opt)

        st.subheader("1. Preprocessing & Word-Level Detection")
        col_orig, col_proc, col_seg = st.columns(3)
        with col_orig:
            st.caption("Original Document")
            st.image(active_image, use_container_width=True)
        with col_proc:
            st.caption(f"Enhanced Document (Skew: {prep_doc.skew_angle:.1f}°)")
            st.image(numpy_to_pil(prep_doc.processed), use_container_width=True)
        with col_seg:
            total_words = sum(len(line.words) for line in ocr_result.lines)
            st.caption(f"Word Bounding Boxes ({total_words} words recognized)")
            word_annotated = segmenter.draw_word_boxes(deskewed_source, ocr_result.lines)
            st.image(numpy_to_pil(word_annotated), use_container_width=True)

        st.markdown("---")
        st.subheader(f"2. Handwriting Recognition ({diag['engine']})")
        st.info(f"ℹ️ **Active Engine**: {diag['engine']} on `{diag['device']}`")

        # Editable Transcript Area
        user_transcript = st.text_area(
            "Recognized Document Text (Editable)",
            value=ocr_result.full_text,
            height=140,
            help="Review or edit recognized handwriting. Downstream classification and extraction update dynamically.",
        )

        # Line-by-line inspection expander
        if ocr_result.lines:
            with st.expander("🔍 Inspect Line-by-Line Predictions"):
                line_table = [
                    {
                        "Line #": f"#{line.line_index + 1}",
                        "Predicted Text": line.text,
                        "Confidence": f"{line.confidence:.1%}",
                        "Bounding Box (X, Y, W, H)": f"{line.bbox}",
                    }
                    for line in ocr_result.lines
                ]
                st.dataframe(pd.DataFrame(line_table), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Stage 3: Document Classification & Information Extraction
        col_class, col_extract = st.columns([1, 2])

        with col_class:
            st.subheader("3. Document Classification")
            class_res = classifier.predict(user_transcript)
            st.metric("Predicted Category", class_res.predicted_class, f"{class_res.confidence:.1%}")
            st.caption(class_res.status_message)

            if class_res.class_probabilities:
                st.caption("Class Probability Distribution:")
                prob_df = pd.DataFrame(
                    list(class_res.class_probabilities.items()),
                    columns=["Class", "Probability"],
                )
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

        with col_extract:
            st.subheader("4. Information Extraction")
            extracted = extractor.extract_all(user_transcript)
            entities_df = extracted.to_dataframe()

            st.caption("Review & Edit Extracted Fields (Add, edit, or remove rows):")
            edited_entities_df = st.data_editor(
                entities_df,
                num_rows="dynamic",
                use_container_width=True,
                key="entity_editor_main",
            )

        st.markdown("---")

        # Stage 4: Persistence and Export
        st.subheader("5. Save & Export")
        col_save, col_exp_json, col_exp_csv, col_exp_pdf = st.columns(4)

        with col_save:
            if st.button("💾 Save Record to SQLite", use_container_width=True):
                doc_record = DocumentRecord(
                    filename=file_name,
                    page_count=loaded_doc.page_count,
                    full_text=user_transcript,
                    document_class=class_res.predicted_class,
                    classification_confidence=class_res.confidence,
                    extracted_fields=extracted.to_dict(),
                )
                record_id = db.insert_document(doc_record)
                st.success(f"Saved to SQLite! (Record #{record_id})")

        with col_exp_json:
            export_payload = {
                "filename": file_name,
                "timestamp": datetime.now().isoformat(),
                "document_class": class_res.predicted_class,
                "confidence": class_res.confidence,
                "transcript": user_transcript,
                "extracted_fields": extracted.to_dict(),
            }
            json_str = ResultExporter.to_json(export_payload)
            st.download_button(
                "⬇️ Download JSON",
                data=json_str,
                file_name=f"{Path(file_name).stem}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_exp_csv:
            csv_str = ResultExporter.to_csv(edited_entities_df)
            st.download_button(
                "⬇️ Download Fields (CSV)",
                data=csv_str,
                file_name=f"{Path(file_name).stem}_fields.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_exp_pdf:
            pdf_bytes = ResultExporter.to_pdf_report(
                filename=file_name,
                document_class=class_res.predicted_class,
                confidence=class_res.confidence,
                full_text=user_transcript,
                extracted_df=edited_entities_df,
            )
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{Path(file_name).stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ---------------------------------------------------------
# TAB 2: SQLite Document History & Archive
# ---------------------------------------------------------
with tab_history:
    st.subheader("SQLite Document Records Archive")
    search_q = st.text_input("🔍 Search archived documents by filename, category, or transcript text:")

    if search_q.strip():
        records = db.search_documents(search_q.strip())
    else:
        records = db.get_all_documents(limit=50)

    if not records:
        st.info("No records found in the database. Process a document and click 'Save Record to SQLite'.")
    else:
        table_rows = []
        for r in records:
            table_rows.append(
                {
                    "ID": r.id,
                    "Filename": r.filename,
                    "Class": r.document_class,
                    "Confidence": f"{r.classification_confidence:.1%}",
                    "Pages": r.page_count,
                    "Saved At": format_timestamp(r.created_at),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Inspect Document Record")
        selected_id = st.selectbox(
            "Select Record ID",
            options=[r.id for r in records],
            format_func=lambda x: f"Record #{x}",
        )

        selected_rec = db.get_document_by_id(selected_id)
        if selected_rec:
            col_det_txt, col_det_act = st.columns([3, 1])
            with col_det_txt:
                st.text_area("Persisted Transcript", value=selected_rec.full_text, height=120, disabled=True)
                st.json(selected_rec.extracted_fields)
            with col_det_act:
                if st.button("🗑️ Delete Record", type="secondary", use_container_width=True):
                    db.delete_document(selected_id)
                    st.success(f"Record #{selected_id} deleted.")
                    st.rerun()


# ---------------------------------------------------------
# TAB 3: Model Diagnostics & Training Instructions
# ---------------------------------------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics & Training Commands")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("### 1. IAM-Line CRNN Recognition Model")
        st.markdown(
            f"""
            - **Dataset:** `Teklia/IAM-line` (Hugging Face)
            - **Architecture:** 7-layer CNN $\\to$ 2-layer BiLSTM $\\to$ CTC Decoder
            - **Checkpoint:** `{iam_predictor.checkpoint_path}`
            - **Status:** `{iam_predictor.status_message}`
            - **Vocabulary:** `{len(iam_predictor.vocabulary) if iam_predictor.vocabulary else 'N/A'}` classes
            """
        )
        st.code("python scripts/train_iam.py --epochs 5 --batch-size 16", language="powershell")
        st.code("python scripts/evaluate_iam.py --max-samples 100", language="powershell")

    with col_d2:
        st.markdown("### 2. Document Classification Model")
        st.markdown(
            f"""
            - **Vectorization:** TF-IDF n-grams (1-2)
            - **Classifier:** Multi-class Logistic Regression
            - **Status:** `{'Trained' if classifier.is_trained else 'Untrained'}`
            """
        )
        st.code("python src/classification/train_classifier.py", language="powershell")

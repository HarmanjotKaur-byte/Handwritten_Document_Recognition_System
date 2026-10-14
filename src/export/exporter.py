"""Export utilities for structured document understanding results.

Converts extracted data into JSON, CSV, and downloadable PDF summary reports.
"""

import io
import json
from typing import Any, Dict, Optional

import pandas as pd


class ResultExporter:
    """Handles serialization of document analysis results into exportable formats."""

    @staticmethod
    def to_json(data: Dict[str, Any], indent: int = 2) -> str:
        """Convert result dictionary to formatted JSON string.

        Args:
            data (Dict[str, Any]): Data payload.
            indent (int): Indentation level.

        Returns:
            str: JSON formatted string.
        """
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        """Convert a Pandas DataFrame of extracted entities to CSV string.

        Args:
            df (pd.DataFrame): Dataframe containing fields.

        Returns:
            str: CSV text string.
        """
        return df.to_csv(index=False)

    @staticmethod
    def to_pdf_report(
        filename: str,
        document_class: str,
        confidence: float,
        full_text: str,
        extracted_df: Optional[pd.DataFrame] = None,
    ) -> bytes:
        """Generate a formatted PDF summary report using ReportLab.

        Args:
            filename (str): Name of the processed document.
            document_class (str): Document classification label.
            confidence (float): Classification confidence score.
            full_text (str): Extracted handwriting transcript.
            extracted_df (Optional[pd.DataFrame]): Table of extracted fields.

        Returns:
            bytes: PDF file content bytes.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        h2_style = styles["Heading2"]
        body_style = styles["Normal"]

        story = []

        # Header Title
        story.append(Paragraph("Handwritten Document Understanding Report", title_style))
        story.append(Spacer(1, 12))

        # Metadata Summary
        meta_text = (
            f"<b>Source File:</b> {filename}<br/>"
            f"<b>Predicted Classification:</b> {document_class} (Confidence: {confidence:.2%})<br/>"
        )
        story.append(Paragraph(meta_text, body_style))
        story.append(Spacer(1, 14))

        # Extracted Entities Table
        story.append(Paragraph("Extracted Information Fields", h2_style))
        story.append(Spacer(1, 8))

        if extracted_df is not None and not extracted_df.empty:
            table_data = [list(extracted_df.columns)]
            for _, row in extracted_df.iterrows():
                table_data.append([str(v) for v in row.values])

            t = Table(table_data, colWidths=[130, 130, 260])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F9FA")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ]
                )
            )
            story.append(t)
        else:
            story.append(Paragraph("<i>No specific structured fields detected.</i>", body_style))

        story.append(Spacer(1, 16))

        # Full Recognized Text
        story.append(Paragraph("Recognized Handwriting Transcript", h2_style))
        story.append(Spacer(1, 8))
        safe_text = (full_text or "No text recognized.").replace("\n", "<br/>")
        story.append(Paragraph(safe_text, body_style))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

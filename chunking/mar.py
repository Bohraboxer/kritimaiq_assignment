import re
import json
from typing import List, Dict
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts clean text from a PDF using marker's PdfConverter.
    """
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    return text


def chunk_text_by_sections(text: str) -> Dict[str, str]:
    """
    Break text into chunks based on sections marked by '## **'.
    """
    sections = text.split('## **')[1:]  # Skip anything before the first marker
    titles = [s.split('\n', 1)[0].strip('**') for s in sections]
    contents = [s.split('\n', 1)[1] if '\n' in s else '' for s in sections]
    return {title: content for title, content in zip(titles, contents)}


def split_into_paragraphs(text: str) -> List[str]:
    """
    Splits section text into paragraphs using double newlines.
    """
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def custom_chunking(pdf_path: str) -> List[Dict]:
    """
    Converts PDF text into paragraph chunks with metadata for vector DB storage.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = []
    section_dict = chunk_text_by_sections(text)

    for section_title, section_body in section_dict.items():
        paragraphs = split_into_paragraphs(section_body)
        for i, para in enumerate(paragraphs):
            chunks.append({
                "id": f"{section_title.replace(' ', '_')}_{i+1}",
                "text": para,
                "metadata": {
                    "section": section_title,
                    "chunk_id": f"{section_title}_{i+1}"
                }
            })
    return chunks
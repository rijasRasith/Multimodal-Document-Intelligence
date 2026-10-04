from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image


def pdf_to_page_images(pdf_path: Path, dpi: int = 150) -> list[Image.Image]:
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def extract_native_text(pdf_path: Path) -> list[str]:
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text.strip())
    doc.close()
    return pages_text


def extract_tables_from_page(page: fitz.Page) -> list[str]:
    tables = []
    try:
        tab = page.find_tables()
        for table in tab.tables:
            rows = table.extract()
            formatted_rows = []
            for row in rows:
                cleaned = [cell.strip() if cell else "" for cell in row]
                formatted_rows.append(" | ".join(cleaned))
            tables.append("\n".join(formatted_rows))
    except Exception:
        pass
    return tables


def extract_all_tables(pdf_path: Path) -> list[list[str]]:
    doc = fitz.open(str(pdf_path))
    all_tables = []
    for page in doc:
        all_tables.append(extract_tables_from_page(page))
    doc.close()
    return all_tables


def generate_page_summary(image: Image.Image, vlm_model, processor) -> str:
    import torch
    from data.schema import EXTRACTION_PROMPT

    summary_prompt = (
        "Provide a concise 2-3 sentence summary of this document page. "
        "Describe the main content, any visible headings, key numbers, and the document type."
    )

    inputs = processor(text=summary_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(vlm_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = vlm_model.generate(**inputs, max_new_tokens=150, do_sample=False)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def ingest_pdf(
    pdf_path: Path,
    doc_id: Optional[str] = None,
    vlm_model=None,
    processor=None,
    dpi: int = 150,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    doc_id = doc_id or str(uuid.uuid4())

    page_images = pdf_to_page_images(pdf_path, dpi=dpi)
    native_texts = extract_native_text(pdf_path)
    all_tables = extract_all_tables(pdf_path)

    nodes = []

    for page_num, (image, page_text, page_tables) in enumerate(
        zip(page_images, native_texts, all_tables), start=1
    ):
        text_chunks = chunk_text(page_text)
        for chunk_idx, chunk in enumerate(text_chunks):
            nodes.append({
                "doc_id": doc_id,
                "source_file": pdf_path.name,
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "content_type": "text",
                "content": chunk,
                "image": None,
            })

        for table_idx, table_text in enumerate(page_tables):
            if table_text.strip():
                nodes.append({
                    "doc_id": doc_id,
                    "source_file": pdf_path.name,
                    "page_number": page_num,
                    "chunk_index": table_idx,
                    "content_type": "table",
                    "content": table_text,
                    "image": None,
                })

        page_summary = ""
        if vlm_model is not None and processor is not None:
            page_summary = generate_page_summary(image, vlm_model, processor)
        else:
            page_summary = page_text[:300] if page_text else f"Page {page_num} of {pdf_path.name}"

        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        nodes.append({
            "doc_id": doc_id,
            "source_file": pdf_path.name,
            "page_number": page_num,
            "chunk_index": 0,
            "content_type": "page_image",
            "content": page_summary,
            "image": img_bytes,
        })

    return nodes


def ingest_image(image_path: Path, doc_id: Optional[str] = None, vlm_model=None, processor=None) -> list[dict]:
    image_path = Path(image_path)
    doc_id = doc_id or str(uuid.uuid4())
    image = Image.open(image_path).convert("RGB")

    page_summary = ""
    if vlm_model is not None and processor is not None:
        page_summary = generate_page_summary(image, vlm_model, processor)
    else:
        page_summary = f"Document image: {image_path.name}"

    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    return [{
        "doc_id": doc_id,
        "source_file": image_path.name,
        "page_number": 1,
        "chunk_index": 0,
        "content_type": "page_image",
        "content": page_summary,
        "image": img_bytes,
    }]

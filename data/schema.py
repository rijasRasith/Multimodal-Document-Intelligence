from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, field_validator
import re


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceSchema(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    invoice_number: Optional[str] = None
    line_items: list[LineItem] = []
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    @field_validator("date", mode="before")
    @classmethod
    def normalize_date(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        patterns = [
            (r"(\d{4})-(\d{2})-(\d{2})", r"\1-\2-\3"),
            (r"(\d{2})/(\d{2})/(\d{4})", r"\3-\2-\1"),
            (r"(\d{2})-(\d{2})-(\d{4})", r"\3-\2-\1"),
        ]
        for pattern, replacement in patterns:
            if re.fullmatch(pattern, v):
                return re.sub(pattern, replacement, v)
        return v

    @field_validator("subtotal", "tax", "total", mode="before")
    @classmethod
    def normalize_amount(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        cleaned = re.sub(r"[^\d.]", "", str(v))
        return round(float(cleaned), 2) if cleaned else None


EXTRACTION_PROMPT = (
    "You are a document understanding assistant. Analyze this invoice or receipt image "
    "and extract the following fields into valid JSON.\n\n"
    "Return ONLY the JSON object with these exact keys:\n"
    "{\n"
    '  "vendor": string or null,\n'
    '  "date": string in YYYY-MM-DD format or null,\n'
    '  "invoice_number": string or null,\n'
    '  "line_items": [\n'
    '    {"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}\n'
    "  ],\n"
    '  "subtotal": number or null,\n'
    '  "tax": number or null,\n'
    '  "total": number or null\n'
    "}\n\n"
    "Rules:\n"
    "- Output null for any field that is not present in the document.\n"
    "- Do not guess or hallucinate values.\n"
    "- Date must be ISO 8601 (YYYY-MM-DD).\n"
    "- Amounts must be plain numbers without currency symbols.\n"
    "- Return only the JSON, no explanation or markdown fences."
)

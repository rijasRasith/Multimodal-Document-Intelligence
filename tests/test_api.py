from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = "cpu"
    model.generate = MagicMock(return_value=MagicMock(shape=[1, 100]))
    return model


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.return_value = {"input_ids": MagicMock(shape=[1, 50])}
    processor.batch_decode = MagicMock(return_value=[
        '{"vendor": "Apex Co.", "date": "2024-01-15", "invoice_number": "INV-123", '
        '"line_items": [], "subtotal": 100.0, "tax": 10.0, "total": 110.0}'
    ])
    return processor


@pytest.fixture
def client(mock_model, mock_processor):
    with (
        patch("api.main._model", mock_model),
        patch("api.main._processor", mock_processor),
        patch("api.main._db_conn", None),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


def _make_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "database_connected" in data


class TestExtractEndpoint:
    def test_extract_requires_api_key(self, client):
        img_bytes = _make_image_bytes()
        response = client.post(
            "/extract",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 403

    def test_extract_with_valid_key(self, client, mock_model, mock_processor):
        img_bytes = _make_image_bytes()

        import torch
        import numpy as np

        input_ids_tensor = torch.zeros((1, 50), dtype=torch.long)
        mock_processor_call_result = {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.ones((1, 50), dtype=torch.long),
        }
        mock_processor.return_value = mock_processor_call_result
        mock_processor.side_effect = None

        generated_ids = torch.zeros((1, 100), dtype=torch.long)
        mock_model.generate.return_value = generated_ids

        valid_json = json.dumps({
            "vendor": "Apex Co.", "date": "2024-01-15",
            "invoice_number": "INV-123", "line_items": [],
            "subtotal": 100.0, "tax": 10.0, "total": 110.0,
        })
        mock_processor.batch_decode.return_value = [valid_json]

        response = client.post(
            "/extract",
            files={"file": ("test.png", img_bytes, "image/png")},
            headers={"X-API-Key": "dev-secret-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "processing_time_ms" in data
        assert "json_valid" in data

    def test_extract_rejects_oversized_file(self, client):
        large_content = b"0" * (25 * 1024 * 1024)
        response = client.post(
            "/extract",
            files={"file": ("big.png", large_content, "image/png")},
            headers={"X-API-Key": "dev-secret-key"},
        )
        assert response.status_code == 413

    def test_extract_rejects_unsupported_type(self, client):
        response = client.post(
            "/extract",
            files={"file": ("doc.txt", b"hello", "text/plain")},
            headers={"X-API-Key": "dev-secret-key"},
        )
        assert response.status_code == 415


class TestSchemaValidation:
    def test_invoice_schema_valid(self):
        from data.schema import InvoiceSchema
        invoice = InvoiceSchema(
            vendor="Apex Co.",
            date="2024-01-15",
            invoice_number="INV-123",
            subtotal=100.0,
            tax=10.0,
            total=110.0,
        )
        assert invoice.vendor == "Apex Co."
        assert invoice.total == 110.0

    def test_invoice_schema_date_normalization(self):
        from data.schema import InvoiceSchema
        invoice = InvoiceSchema(date="15/01/2024")
        assert invoice.date == "2024-01-15"

    def test_invoice_schema_amount_normalization(self):
        from data.schema import InvoiceSchema
        invoice = InvoiceSchema(total="$1,234.56")
        assert invoice.total == 1234.56

    def test_invoice_schema_allows_none_fields(self):
        from data.schema import InvoiceSchema
        invoice = InvoiceSchema()
        assert invoice.vendor is None
        assert invoice.total is None
        assert invoice.line_items == []


class TestFeedbackEndpoint:
    def test_feedback_requires_db(self, client):
        response = client.post(
            "/feedback",
            json={
                "request_id": "abc-123",
                "endpoint": "/extract",
                "rating": 1,
            },
            headers={"X-API-Key": "dev-secret-key"},
        )
        assert response.status_code == 503

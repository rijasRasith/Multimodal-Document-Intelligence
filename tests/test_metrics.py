from __future__ import annotations

import json
import pytest

from evaluation.metrics import (
    compute_field_f1,
    field_exact_match,
    fuzzy_match,
    match_line_items,
    normalize_amount,
    normalize_date,
    normalize_string,
    parse_model_output,
)


class TestNormalization:
    def test_normalize_string_strips_and_lowercases(self):
        assert normalize_string("  Hello World  ") == "hello world"

    def test_normalize_string_collapses_whitespace(self):
        assert normalize_string("hello  world") == "hello world"

    def test_normalize_string_none(self):
        assert normalize_string(None) == ""

    def test_normalize_amount_strips_currency(self):
        assert normalize_amount("$1,234.56") == "1234.56"

    def test_normalize_amount_plain_float(self):
        assert normalize_amount(100.5) == "100.50"

    def test_normalize_amount_none(self):
        assert normalize_amount(None) == ""

    def test_normalize_date_iso(self):
        assert normalize_date("2024-01-15") == "2024-01-15"

    def test_normalize_date_dd_mm_yyyy_slash(self):
        assert normalize_date("15/01/2024") == "2024-01-15"

    def test_normalize_date_dd_mm_yyyy_dash(self):
        assert normalize_date("15-01-2024") == "2024-01-15"

    def test_normalize_date_none(self):
        assert normalize_date(None) == ""


class TestFieldExactMatch:
    def test_vendor_match(self):
        assert field_exact_match("Apex Co.", "apex co.", "vendor") is True

    def test_vendor_mismatch(self):
        assert field_exact_match("Apex Co.", "Beta Co.", "vendor") is False

    def test_total_match_with_symbol(self):
        assert field_exact_match("$100.50", 100.5, "total") is True

    def test_total_mismatch(self):
        assert field_exact_match(99.99, 100.00, "total") is False

    def test_date_match_different_formats(self):
        assert field_exact_match("15/01/2024", "2024-01-15", "date") is True

    def test_both_none(self):
        assert field_exact_match(None, None, "vendor") is True

    def test_pred_none_gt_has_value(self):
        assert field_exact_match(None, "Apex Co.", "vendor") is False


class TestFuzzyMatch:
    def test_high_similarity(self):
        assert fuzzy_match("Apex Supplies Company", "Apex Supplies Co") is True

    def test_low_similarity(self):
        assert fuzzy_match("Apex", "Beta Corp", threshold=0.85) is False

    def test_both_none(self):
        assert fuzzy_match(None, None) is True

    def test_one_none(self):
        assert fuzzy_match(None, "Some Value") is False


class TestMatchLineItems:
    def test_perfect_match(self):
        pred = [{"description": "Printer Paper", "quantity": 2.0, "unit_price": 8.99, "total": 17.98}]
        gt = [{"description": "Printer Paper", "quantity": 2.0, "unit_price": 8.99, "total": 17.98}]
        tp, fp, fn = match_line_items(pred, gt)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_empty_prediction(self):
        gt = [{"description": "Printer Paper", "quantity": 1.0, "unit_price": 8.99, "total": 8.99}]
        tp, fp, fn = match_line_items([], gt)
        assert tp == 0
        assert fp == 0
        assert fn == 1

    def test_extra_prediction(self):
        pred = [
            {"description": "Printer Paper", "quantity": 1.0, "unit_price": 8.99, "total": 8.99},
            {"description": "Stapler", "quantity": 1.0, "unit_price": 12.99, "total": 12.99},
        ]
        gt = [{"description": "Printer Paper", "quantity": 1.0, "unit_price": 8.99, "total": 8.99}]
        tp, fp, fn = match_line_items(pred, gt)
        assert tp == 1
        assert fp == 1
        assert fn == 0


class TestComputeFieldF1:
    def test_perfect_predictions(self):
        gt = {"vendor": "Apex Co.", "date": "2024-01-15", "invoice_number": "INV-123",
              "subtotal": 100.0, "tax": 10.0, "total": 110.0, "line_items": []}
        metrics = compute_field_f1([gt], [gt])
        assert metrics["macro_f1"] == 1.0
        assert metrics["json_validity_rate"] == 1.0

    def test_all_wrong(self):
        pred = {"vendor": "Wrong", "date": "2000-01-01", "invoice_number": "BAD",
                "subtotal": 0.0, "tax": 0.0, "total": 0.0, "line_items": []}
        gt = {"vendor": "Apex Co.", "date": "2024-01-15", "invoice_number": "INV-123",
              "subtotal": 100.0, "tax": 10.0, "total": 110.0, "line_items": []}
        metrics = compute_field_f1([pred], [gt])
        assert metrics["macro_f1"] < 0.5

    def test_invalid_json_prediction(self):
        gt = {"vendor": "Apex Co.", "date": "2024-01-15", "invoice_number": "INV-123",
              "subtotal": 100.0, "tax": 10.0, "total": 110.0, "line_items": []}
        metrics = compute_field_f1([None], [gt])
        assert metrics["json_validity_rate"] == 0.0

    def test_empty_inputs(self):
        metrics = compute_field_f1([], [])
        assert metrics["macro_f1"] == 0.0


class TestParseModelOutput:
    def test_plain_json(self):
        raw = '{"vendor": "Apex", "total": 100.0}'
        result = parse_model_output(raw)
        assert result == {"vendor": "Apex", "total": 100.0}

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"vendor": "Apex"}\n```'
        result = parse_model_output(raw)
        assert result == {"vendor": "Apex"}

    def test_json_with_preamble(self):
        raw = 'Here is the result:\n{"vendor": "Apex", "total": 50.0}'
        result = parse_model_output(raw)
        assert result == {"vendor": "Apex", "total": 50.0}

    def test_malformed_json(self):
        raw = "This is not JSON at all."
        result = parse_model_output(raw)
        assert result is None

    def test_empty_string(self):
        result = parse_model_output("")
        assert result is None

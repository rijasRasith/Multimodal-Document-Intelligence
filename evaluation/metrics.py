from __future__ import annotations

import json
import re
import difflib
from typing import Any


SCALAR_FIELDS = ["vendor", "date", "invoice_number", "subtotal", "tax", "total"]
NUMERIC_FIELDS = {"subtotal", "tax", "total"}
DATE_FIELD = "date"


def normalize_string(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_amount(value: Any) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return f"{float(cleaned):.2f}"
    except ValueError:
        return ""


def normalize_date(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", r"\1-\2-\3"),
        (r"(\d{2})/(\d{2})/(\d{4})", r"\3-\2-\1"),
        (r"(\d{2})-(\d{2})-(\d{4})", r"\3-\2-\1"),
    ]
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, value):
            return re.sub(pattern, replacement, value)
    return normalize_string(value)


def _normalize_field(field: str, value: Any) -> str:
    if field in NUMERIC_FIELDS:
        return normalize_amount(value)
    if field == DATE_FIELD:
        return normalize_date(value)
    return normalize_string(str(value) if value is not None else None)


def field_exact_match(pred_val: Any, gt_val: Any, field: str) -> bool:
    pred_norm = _normalize_field(field, pred_val)
    gt_norm = _normalize_field(field, gt_val)
    return pred_norm == gt_norm and (pred_norm != "" or gt_norm == "")


def fuzzy_match(pred_val: Any, gt_val: Any, threshold: float = 0.85) -> bool:
    pred_norm = normalize_string(str(pred_val) if pred_val is not None else None)
    gt_norm = normalize_string(str(gt_val) if gt_val is not None else None)
    if not pred_norm and not gt_norm:
        return True
    if not pred_norm or not gt_norm:
        return False
    ratio = difflib.SequenceMatcher(None, pred_norm, gt_norm).ratio()
    return ratio >= threshold


def match_line_items(pred_items: list[dict], gt_items: list[dict]) -> tuple[int, int, int]:
    if not gt_items:
        return 0, len(pred_items), 0

    matched_gt = set()
    true_positives = 0

    for pred in pred_items:
        pred_desc = normalize_string(pred.get("description", ""))
        best_ratio = 0.0
        best_idx = -1

        for i, gt in enumerate(gt_items):
            if i in matched_gt:
                continue
            gt_desc = normalize_string(gt.get("description", ""))
            ratio = difflib.SequenceMatcher(None, pred_desc, gt_desc).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_ratio >= 0.7 and best_idx >= 0:
            gt = gt_items[best_idx]
            matched_gt.add(best_idx)

            price_match = normalize_amount(pred.get("unit_price")) == normalize_amount(gt.get("unit_price"))
            qty_match = normalize_amount(pred.get("quantity")) == normalize_amount(gt.get("quantity"))
            item_match = price_match or qty_match
            if item_match:
                true_positives += 1

    false_positives = len(pred_items) - true_positives
    false_negatives = len(gt_items) - len(matched_gt)
    return true_positives, false_positives, false_negatives


def compute_field_f1(predictions: list[dict], ground_truths: list[dict]) -> dict:
    field_stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in SCALAR_FIELDS}
    field_stats["line_items"] = {"tp": 0, "fp": 0, "fn": 0}

    json_valid_count = 0
    exact_doc_match_count = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_valid = isinstance(pred, dict)
        if pred_valid:
            json_valid_count += 1

        doc_exact = True
        for field in SCALAR_FIELDS:
            pred_val = pred.get(field) if pred_valid else None
            gt_val = gt.get(field)

            is_match = field_exact_match(pred_val, gt_val, field)
            if is_match:
                field_stats[field]["tp"] += 1
            else:
                doc_exact = False
                if pred_val is not None and _normalize_field(field, pred_val):
                    field_stats[field]["fp"] += 1
                if gt_val is not None and _normalize_field(field, gt_val):
                    field_stats[field]["fn"] += 1

        pred_items = pred.get("line_items", []) if pred_valid else []
        gt_items = gt.get("line_items", [])
        tp, fp, fn = match_line_items(pred_items, gt_items)
        field_stats["line_items"]["tp"] += tp
        field_stats["line_items"]["fp"] += fp
        field_stats["line_items"]["fn"] += fn
        if tp == len(gt_items) and fp == 0:
            pass
        else:
            doc_exact = False

        if doc_exact:
            exact_doc_match_count += 1

    results = {}
    f1_scores = []
    for field, stats in field_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[field] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        f1_scores.append(f1)

    n = len(predictions)
    results["macro_f1"] = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
    results["json_validity_rate"] = round(json_valid_count / n, 4) if n > 0 else 0.0
    results["exact_document_match_rate"] = round(exact_doc_match_count / n, 4) if n > 0 else 0.0

    return results


def parse_model_output(raw_output: str) -> dict | None:
    raw_output = raw_output.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
    if fence_match:
        raw_output = fence_match.group(1).strip()

    brace_match = re.search(r"\{[\s\S]*\}", raw_output)
    if brace_match:
        raw_output = brace_match.group(0)

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return None


def print_metrics_table(metrics: dict, model_name: str = "Model") -> None:
    print(f"\n{'='*60}")
    print(f"  Results: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")

    field_order = SCALAR_FIELDS + ["line_items"]
    for field in field_order:
        if field in metrics:
            m = metrics[field]
            print(f"  {field:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print(f"  {'-'*50}")
    print(f"  {'Macro F1':<20} {metrics.get('macro_f1', 0.0):>32.4f}")
    print(f"  {'JSON Validity':<20} {metrics.get('json_validity_rate', 0.0):>32.4f}")
    print(f"  {'Exact Doc Match':<20} {metrics.get('exact_document_match_rate', 0.0):>32.4f}")
    print(f"{'='*60}\n")

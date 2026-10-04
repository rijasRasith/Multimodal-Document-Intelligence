from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import mlflow
import psycopg2


WINDOW_SECONDS = 300
MAX_HISTORY = 1000

_latency_window: deque = deque(maxlen=MAX_HISTORY)
_error_window: deque = deque(maxlen=MAX_HISTORY)
_invalid_json_window: deque = deque(maxlen=MAX_HISTORY)
_retrieval_score_window: deque = deque(maxlen=MAX_HISTORY)


def record_request(
    latency_ms: float,
    is_error: bool = False,
    is_json_invalid: bool = False,
    retrieval_score: Optional[float] = None,
) -> None:
    ts = time.time()
    _latency_window.append((ts, latency_ms))
    _error_window.append((ts, 1 if is_error else 0))
    _invalid_json_window.append((ts, 1 if is_json_invalid else 0))
    if retrieval_score is not None:
        _retrieval_score_window.append((ts, retrieval_score))


def _window_values(window: deque, seconds: int = WINDOW_SECONDS) -> list:
    cutoff = time.time() - seconds
    return [v for ts, v in window if ts >= cutoff]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def get_live_metrics() -> dict:
    latencies = _window_values(_latency_window)
    errors = _window_values(_error_window)
    invalid_jsons = _window_values(_invalid_json_window)
    retrieval_scores = _window_values(_retrieval_score_window)

    n = len(latencies)
    error_count = sum(errors)
    invalid_json_count = sum(invalid_jsons)

    return {
        "window_seconds": WINDOW_SECONDS,
        "request_count": n,
        "p50_latency_ms": round(_percentile(latencies, 50), 2),
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "error_rate": round(error_count / max(n, 1), 4),
        "invalid_json_rate": round(invalid_json_count / max(n, 1), 4),
        "avg_retrieval_score": round(sum(retrieval_scores) / max(len(retrieval_scores), 1), 4),
    }


ALERT_THRESHOLDS = {
    "invalid_json_rate": 0.15,
    "error_rate": 0.10,
    "p95_latency_ms": 30000,
}


def check_alerts(metrics: dict) -> list[str]:
    alerts = []
    if metrics["invalid_json_rate"] > ALERT_THRESHOLDS["invalid_json_rate"]:
        alerts.append(
            f"ALERT: invalid_json_rate {metrics['invalid_json_rate']:.1%} exceeds threshold "
            f"{ALERT_THRESHOLDS['invalid_json_rate']:.1%}. "
            "Action: inspect recent predictions, check model output format."
        )
    if metrics["error_rate"] > ALERT_THRESHOLDS["error_rate"]:
        alerts.append(
            f"ALERT: error_rate {metrics['error_rate']:.1%} exceeds threshold "
            f"{ALERT_THRESHOLDS['error_rate']:.1%}. "
            "Action: check API logs, verify model and database connectivity."
        )
    if metrics["p95_latency_ms"] > ALERT_THRESHOLDS["p95_latency_ms"]:
        alerts.append(
            f"ALERT: p95_latency {metrics['p95_latency_ms']:.0f}ms exceeds threshold "
            f"{ALERT_THRESHOLDS['p95_latency_ms']:.0f}ms. "
            "Action: investigate model load, consider batching or hardware upgrade."
        )
    return alerts


def log_metrics_to_mlflow(metrics: dict, run_name: str = "monitoring") -> None:
    mlflow.set_experiment("production_monitoring")
    with mlflow.start_run(run_name=run_name):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)


def get_feedback_summary(conn, limit: int = 100) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT rating, COUNT(*) as cnt
            FROM feedback
            GROUP BY rating
            ORDER BY rating;
            """,
        )
        rows = cur.fetchall()

    counts = {r[0]: r[1] for r in rows}
    total = sum(counts.values())
    return {
        "total_feedback": total,
        "thumbs_up": counts.get(1, 0),
        "neutral": counts.get(0, 0),
        "thumbs_down": counts.get(-1, 0),
        "positive_rate": round(counts.get(1, 0) / max(total, 1), 4),
    }


def get_review_queue(conn, limit: int = 50) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT request_id, endpoint, rating, corrected_fields, original_output, created_at
            FROM feedback
            WHERE rating = -1 AND corrected_fields IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %s;
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [
        {
            "request_id": r[0],
            "endpoint": r[1],
            "rating": r[2],
            "corrected_fields": r[3],
            "original_output": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]

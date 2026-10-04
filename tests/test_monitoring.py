from __future__ import annotations

import json
import time
import pytest

from api.monitoring import (
    check_alerts,
    get_live_metrics,
    record_request,
    ALERT_THRESHOLDS,
    _latency_window,
    _error_window,
    _invalid_json_window,
    _retrieval_score_window,
)


@pytest.fixture(autouse=True)
def clear_windows():
    _latency_window.clear()
    _error_window.clear()
    _invalid_json_window.clear()
    _retrieval_score_window.clear()
    yield
    _latency_window.clear()
    _error_window.clear()
    _invalid_json_window.clear()
    _retrieval_score_window.clear()


class TestRecordAndMetrics:
    def test_empty_windows_return_zero_metrics(self):
        metrics = get_live_metrics()
        assert metrics["p50_latency_ms"] == 0.0
        assert metrics["error_rate"] == 0.0
        assert metrics["invalid_json_rate"] == 0.0

    def test_latency_recorded(self):
        record_request(latency_ms=250.0)
        record_request(latency_ms=500.0)
        metrics = get_live_metrics()
        assert metrics["p50_latency_ms"] > 0
        assert metrics["request_count"] == 2

    def test_error_rate_calculation(self):
        record_request(latency_ms=100.0, is_error=True)
        record_request(latency_ms=100.0, is_error=True)
        record_request(latency_ms=100.0, is_error=False)
        record_request(latency_ms=100.0, is_error=False)
        metrics = get_live_metrics()
        assert abs(metrics["error_rate"] - 0.5) < 1e-4

    def test_invalid_json_rate(self):
        for _ in range(3):
            record_request(latency_ms=100.0, is_json_invalid=True)
        for _ in range(7):
            record_request(latency_ms=100.0, is_json_invalid=False)
        metrics = get_live_metrics()
        assert abs(metrics["invalid_json_rate"] - 0.3) < 1e-4

    def test_retrieval_score_recorded(self):
        record_request(latency_ms=100.0, retrieval_score=0.85)
        record_request(latency_ms=100.0, retrieval_score=0.75)
        metrics = get_live_metrics()
        assert abs(metrics["avg_retrieval_score"] - 0.80) < 1e-4


class TestAlerts:
    def test_no_alerts_when_below_threshold(self):
        for _ in range(10):
            record_request(latency_ms=500.0)
        metrics = get_live_metrics()
        alerts = check_alerts(metrics)
        assert len(alerts) == 0

    def test_invalid_json_alert_fires(self):
        metrics_over = {
            "invalid_json_rate": ALERT_THRESHOLDS["invalid_json_rate"] + 0.01,
            "error_rate": 0.0,
            "p95_latency_ms": 0.0,
        }
        alerts = check_alerts(metrics_over)
        assert any("invalid_json_rate" in a for a in alerts)

    def test_error_rate_alert_fires(self):
        metrics_over = {
            "invalid_json_rate": 0.0,
            "error_rate": ALERT_THRESHOLDS["error_rate"] + 0.01,
            "p95_latency_ms": 0.0,
        }
        alerts = check_alerts(metrics_over)
        assert any("error_rate" in a for a in alerts)

    def test_latency_alert_fires(self):
        metrics_over = {
            "invalid_json_rate": 0.0,
            "error_rate": 0.0,
            "p95_latency_ms": ALERT_THRESHOLDS["p95_latency_ms"] + 1,
        }
        alerts = check_alerts(metrics_over)
        assert any("p95_latency" in a for a in alerts)

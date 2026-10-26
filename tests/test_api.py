"""Tests for FastAPI backend in qpred/api.py.

Verifies health, system, runs, registries, benchmarks, checkpoints, SPA, and comparison.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qpred.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_system_endpoint():
    response = client.get("/api/system")
    assert response.status_code == 200
    data = response.json()
    assert "cuda_available" in data
    assert "pytorch" in data
    assert "cached_models" in data


def test_models_registry_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    model_ids = [m["id"] for m in data]
    assert "qpredrnn" in model_ids
    assert "predrnn_v2" in model_ids
    assert "convlstm" in model_ids
    assert "persistence" in model_ids


def test_experiments_registry_endpoint():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 10
    qpred_50 = next((r for r in data if r["id"] == "kth_qpredrnn_local_50ep"), None)
    assert qpred_50 is not None
    assert qpred_50["category"] == "primary"


def test_checkpoints_inventory_endpoint():
    response = client.get("/api/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 24


def test_benchmarks_endpoint():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    model_ids = [m["model_id"] for m in data]
    assert "qpredrnn_50ep" in model_ids
    assert "persistence" in model_ids


def test_runs_endpoint():
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_datasets_endpoint():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    dataset_ids = [d["id"] for d in data]
    assert "kth" in dataset_ids or "toy" in dataset_ids


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "mse" in data["metrics"]
    assert "ssim" in data["metrics"]


def test_invalid_run_not_found():
    response = client.get("/api/runs/non_existent_run_12345")
    assert response.status_code == 404


def test_predict_validation_error():
    # Bad checkpoint value should raise 422
    response = client.post(
        "/api/predict",
        json={
            "run_id": "test_run",
            "checkpoint": "invalid_value",
            "dataset": "kth",
            "split": "test",
            "clip_index": 0,
        },
    )
    assert response.status_code == 422


def test_predict_analytical_baseline():
    response = client.post(
        "/api/predict",
        json={
            "run_id": "baseline_persistence",
            "checkpoint": "best",
            "dataset": "kth",
            "split": "test",
            "clip_index": 0,
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "pred_frames" in data
    assert len(data["pred_frames"]) == 10


def test_compare_endpoint():
    response = client.post(
        "/api/compare",
        json={
            "models": [
                {"run_id": "baseline_persistence", "checkpoint": "best"},
                {"run_id": "baseline_mean", "checkpoint": "best"},
            ],
            "dataset": "kth",
            "split": "test",
            "clip_index": 0,
            "device": "cpu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 2
    assert data["models"][0]["success"] is True


def test_spa_index_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "QPredRNN" in response.text


def test_unknown_api_returns_404():
    response = client.get("/api/nonexistent_route")
    assert response.status_code == 404

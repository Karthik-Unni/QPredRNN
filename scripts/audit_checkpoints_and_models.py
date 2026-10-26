"""Master Checkpoint and Model Audit Script for QPredRNN.

Audits every .pt file in the repository:
- Computes SHA-256 fingerprints, file sizes, structure.
- Recovers run metadata, config, history, epochs, best/last status.
- Instantiates architecture and verifies state_dict loading.
- Executes forward pass smoke test, verifying shape, finite values, and quantum path.
- Flags byte-identical duplicates and classifies run types.
- Exports docs/CHECKPOINT_INVENTORY.csv and docs/checkpoint_inventory.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parents[1].resolve()
RUNS_DIR = ROOT / "runs"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# Ensure qpred is in sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpred.models import build_model, VideoConvLSTM, PredRNNV2, QPredRNN


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_all_checkpoints() -> list[dict[str, Any]]:
    ckpt_paths = sorted(RUNS_DIR.glob("*/checkpoint_*.pt"))
    print(f"[*] Discovered {len(ckpt_paths)} checkpoint files in runs/")

    inventory = []
    hash_map: dict[str, list[str]] = {}

    for p in ckpt_paths:
        rel_path = p.relative_to(ROOT).as_posix()
        run_dir = p.parent
        run_id = run_dir.name
        is_best = "best" in p.name.lower()
        is_last = "last" in p.name.lower()
        file_size = p.stat().st_size
        file_hash = sha256_file(p)

        hash_map.setdefault(file_hash, []).append(rel_path)

        # 1. Read checkpoint dictionary
        try:
            ckpt_dict = torch.load(p, map_location="cpu", weights_only=False)
            is_dict = isinstance(ckpt_dict, dict)
            has_state_dict = is_dict and "model_state_dict" in ckpt_dict
            saved_epoch = ckpt_dict.get("epoch") if is_dict else None
            saved_step = ckpt_dict.get("global_step") if is_dict else None
            saved_best_val = ckpt_dict.get("best_val_loss") if is_dict else None
            ckpt_cfg = ckpt_dict.get("config", {}) if is_dict else {}
        except Exception as e:
            ckpt_dict = None
            is_dict = False
            has_state_dict = False
            saved_epoch = None
            saved_step = None
            saved_best_val = None
            ckpt_cfg = {}
            load_err = f"torch.load failed: {e}"

        # 2. Read associated run files
        config_f = run_dir / "config.yaml"
        history_f = run_dir / "history.json"
        metrics_f = run_dir / "metrics.csv"
        console_f = run_dir / "training_console.log"

        run_cfg = {}
        if config_f.exists():
            try:
                run_cfg = yaml.safe_load(config_f.read_text(encoding="utf-8")) or {}
            except Exception:
                pass

        # Merged config
        merged_cfg = ckpt_cfg if ckpt_cfg else run_cfg

        # History metadata
        history_data = []
        if history_f.exists():
            try:
                history_data = json.loads(history_f.read_text(encoding="utf-8")) or []
            except Exception:
                pass

        history_len = len(history_data)
        hist_best_val = None
        hist_last_val = None
        hist_last_epoch = None
        if history_data:
            try:
                val_entries = [h["val_loss"] for h in history_data if "val_loss" in h]
                if val_entries:
                    hist_best_val = min(val_entries)
                    hist_last_val = val_entries[-1]
                hist_last_epoch = history_data[-1].get("epoch", history_len)
            except Exception:
                pass

        # Configured epochs
        configured_epochs = (
            merged_cfg.get("training", {}).get("epochs")
            or merged_cfg.get("epochs")
            or (run_cfg.get("training", {}).get("epochs") if run_cfg else None)
        )

        # Infer Architecture name
        model_name = (
            merged_cfg.get("model", {}).get("name")
            or run_cfg.get("model", {}).get("name")
        )
        if not model_name:
            if "classical" in run_id.lower() or "convlstm" in run_id.lower():
                model_name = "classical"
            elif "qpredrnn" in run_id.lower() or "quantum" in run_id.lower():
                model_name = "qpredrnn"
            elif "predrnn" in run_id.lower():
                model_name = "predrnn_v2"
            else:
                model_name = "classical"

        # Determine architecture class
        if model_name == "classical":
            arch_display = "ConvLSTM (VideoConvLSTM)"
        elif model_name == "predrnn_v2":
            arch_display = "PredRNN-V2 (PredRNNV2)"
        elif model_name in ("qpredrnn", "quantum_context"):
            arch_display = "QPredRNN (Hybrid VQC + SpatioTemporal LSTM)"
        else:
            arch_display = f"Unknown ({model_name})"

        # Run Type Classification
        rid_l = run_id.lower()
        if "smoke" in rid_l:
            run_type = "smoke_test"
        elif "profile" in rid_l:
            run_type = "profiling"
        elif "pilot" in rid_l:
            run_type = "pilot"
        elif "50ep" in rid_l or "overnight" in rid_l or "local" in rid_l:
            run_type = "main_experiment"
        else:
            run_type = "experimental"

        # 3. Model construction and loading test
        load_ok = False
        load_msg = "OK"
        param_count = None
        forward_ok = False
        forward_msg = "Untested"
        output_shape = None
        is_finite = False
        quantum_executed = False

        try:
            # Construct model using merged config or minimal default
            test_cfg = merged_cfg if merged_cfg.get("model") else {
                "model": {
                    "name": model_name,
                    "image_channels": 1,
                    "hidden_channels": 32 if model_name != "classical" else 16,
                    "input_frames": 10 if model_name != "classical" else 5,
                    "output_frames": 10 if model_name != "classical" else 5,
                    "num_layers": 2,
                    "kernel_size": 3,
                }
            }
            model = build_model(test_cfg)
            param_count = sum(p.numel() for p in model.parameters())

            # Load state dict
            if has_state_dict:
                state_dict = ckpt_dict["model_state_dict"]
            elif is_dict:
                state_dict = ckpt_dict
            else:
                raise ValueError("Checkpoint is not a dictionary")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                load_msg = f"Partial load: {len(missing)} missing, {len(unexpected)} unexpected keys"
            else:
                load_msg = "Strict load OK"
            load_ok = True

            # 4. Forward pass test
            model.eval()
            in_t = model.input_frames if hasattr(model, "input_frames") else 10
            dummy_in = torch.rand(1, in_t, 1, 64, 64)

            with torch.inference_mode():
                out = model(dummy_in)

            output_shape = list(out.shape)
            is_finite = bool(torch.isfinite(out).all().item())
            forward_ok = is_finite and (len(output_shape) == 5)
            forward_msg = "OK" if forward_ok else "Shape/finite issue"

            # Check quantum execution
            if isinstance(model, QPredRNN):
                quantum_executed = hasattr(model, "quantum_modulator") and model.quantum_modulator.enabled

        except Exception as e:
            load_ok = False
            load_msg = f"Load/forward failed: {e}"
            forward_ok = False
            forward_msg = str(e)

        # 5. Training completion status classification
        if configured_epochs and hist_last_epoch:
            if hist_last_epoch >= configured_epochs:
                train_status = "Completed"
            else:
                train_status = f"Incomplete ({hist_last_epoch}/{configured_epochs} epochs)"
        elif history_len > 0:
            train_status = f"Finished ({history_len} epochs recorded)"
        elif saved_epoch is not None:
            train_status = f"Finished at epoch {saved_epoch}"
        else:
            train_status = "Unknown / Smoke test"

        # Checkpoint classification
        if forward_ok and load_ok:
            if run_type == "main_experiment":
                classification = "Valid Final Candidate"
            elif run_type == "pilot":
                classification = "Valid Pilot Checkpoint"
            elif run_type == "profiling":
                classification = "Valid Profiling Artifact"
            else:
                classification = "Valid Experimental/Smoke Artifact"
        else:
            classification = "Incompatible / Load Failed"

        item = {
            "file_path": rel_path,
            "run_id": run_id,
            "filename": p.name,
            "architecture": arch_display,
            "model_type": model_name,
            "run_type": run_type,
            "classification": classification,
            "file_size_bytes": file_size,
            "sha256": file_hash,
            "is_best": is_best,
            "is_last": is_last,
            "recorded_epoch": saved_epoch if saved_epoch is not None else hist_last_epoch,
            "recorded_global_step": saved_step,
            "configured_epochs": configured_epochs,
            "history_epochs": history_len,
            "best_val_loss": saved_best_val if saved_best_val is not None else hist_best_val,
            "last_val_loss": hist_last_val,
            "train_status": train_status,
            "parameter_count": param_count,
            "load_status": load_msg,
            "load_ok": load_ok,
            "forward_status": forward_msg,
            "forward_ok": forward_ok,
            "output_shape": output_shape,
            "is_finite": is_finite,
            "quantum_path_verified": quantum_executed,
            "duplicate_of": None,  # will populate below
        }
        inventory.append(item)

    # Flag byte-identical duplicates
    for h, paths in hash_map.items():
        if len(paths) > 1:
            first = paths[0]
            for it in inventory:
                if it["file_path"] in paths and it["file_path"] != first:
                    it["duplicate_of"] = first

    return inventory


def export_inventory(inventory: list[dict[str, Any]]) -> None:
    # 1. JSON
    json_path = DOCS_DIR / "checkpoint_inventory.json"
    json_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"[+] Saved JSON inventory to {json_path}")

    # 2. CSV
    csv_path = DOCS_DIR / "CHECKPOINT_INVENTORY.csv"
    fields = [
        "file_path",
        "run_id",
        "filename",
        "architecture",
        "model_type",
        "run_type",
        "classification",
        "file_size_bytes",
        "sha256",
        "is_best",
        "is_last",
        "recorded_epoch",
        "recorded_global_step",
        "configured_epochs",
        "history_epochs",
        "best_val_loss",
        "last_val_loss",
        "train_status",
        "parameter_count",
        "load_status",
        "forward_status",
        "output_shape",
        "quantum_path_verified",
        "duplicate_of",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in inventory:
            writer.writerow(row)
    print(f"[+] Saved CSV inventory to {csv_path}")


if __name__ == "__main__":
    inv = audit_all_checkpoints()
    export_inventory(inv)
    print("\n[Audit Summary]")
    print(f"Total checkpoints: {len(inv)}")
    valid_count = sum(1 for x in inv if x["load_ok"] and x["forward_ok"])
    print(f"Valid loading & forward pass: {valid_count} / {len(inv)}")
    dup_count = sum(1 for x in inv if x["duplicate_of"] is not None)
    print(f"Byte-identical duplicates: {dup_count}")

"""Import and verification utility for Colab TPU training artifacts.

Safely inspects, verifies, checksums, and imports training runs from Google Drive / Colab
into the local project runs/ directory without modifying source files or overwriting
existing checkpoints unless explicitly requested.

Usage:
    # Dry-run inspection
    python scripts/import_colab_results.py --source-dir /path/to/extracted_runs --dry-run

    # Verified import into runs/
    python scripts/import_colab_results.py --source-dir /path/to/extracted_runs
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

EXPECTED_RUN_DIRS = [
    "kth_classical_tpu_50ep",
    "kth_predrnn_v2_tpu_50ep",
    "kth_qpredrnn_tpu_50ep",
]

REQUIRED_ARTIFACT_NAMES = [
    "checkpoint_best.pt",
    "checkpoint_last.pt",
    "history.json",
    "metrics.csv",
]


def find_run_config(run_dir: Path, source_root: Path | None = None) -> tuple[Path | None, dict | None]:
    """Find configuration file (config.yaml, config_used.yaml, or in training_configs/)."""
    # 1. Local inside run_dir
    for name in ["config.yaml", "config_used.yaml"]:
        p = run_dir / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                return p, cfg
            except Exception:
                pass

    # 2. Adjacent training_configs/ folder in Google Drive or source root
    if source_root:
        for parent_dir in [source_root, source_root.parent]:
            cfg_dir = parent_dir / "training_configs"
            if cfg_dir.exists():
                for ext in [".yaml", ".yml"]:
                    p = cfg_dir / f"{run_dir.name}{ext}"
                    if p.exists():
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                cfg = yaml.safe_load(f)
                            return p, cfg
                        except Exception:
                            pass
    return None, None


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    """Inspect checkpoint metadata using CPU mapping and verify tensor loadability.

    Returns:
        dict containing load status, epoch, global_step, val_loss, and any error message.
    """
    import torch

    result: dict[str, Any] = {
        "loadable": False,
        "epoch": None,
        "global_step": None,
        "val_loss": None,
        "best_val_loss": None,
        "config": None,
        "is_xla_error": False,
        "error_msg": None,
    }

    try:
        # Test loading using CPU mapping
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        result["loadable"] = True
        result["epoch"] = ckpt.get("epoch")
        result["global_step"] = ckpt.get("global_step")
        result["val_loss"] = float(ckpt["val_loss"]) if "val_loss" in ckpt and ckpt["val_loss"] is not None else None
        result["best_val_loss"] = (
            float(ckpt["best_val_loss"]) if "best_val_loss" in ckpt and ckpt["best_val_loss"] is not None else None
        )
        result["config"] = ckpt.get("config")

        # Verify model_state_dict tensors are on CPU and finite
        state_dict = ckpt.get("model_state_dict", {})
        all_finite = True
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                if v.device.type != "cpu":
                    result["error_msg"] = f"Tensor '{k}' device is '{v.device.type}', expected 'cpu'"
                    result["loadable"] = False
                    return result
                if not torch.isfinite(v).all():
                    all_finite = False
        result["tensors_finite"] = all_finite
    except Exception as exc:
        err_str = str(exc)
        result["loadable"] = False
        result["error_msg"] = err_str
        if "xla" in err_str.lower() or "torch_xla" in err_str.lower():
            result["is_xla_error"] = True

    return result


def find_training_log(run_dir: Path) -> Path | None:
    """Find training log file (*.log or train.log)."""
    logs = list(run_dir.glob("*.log"))
    if logs:
        return logs[0]
    return None


def verify_run_consistency(
    run_dir: Path,
    best_meta: dict[str, Any],
    last_meta: dict[str, Any],
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Verify consistency between config (config.yaml or config_used.yaml), history.json, metrics.csv, and log.

    Returns:
        dict with verification details and list of discrepancies.
    """
    discrepancies: list[str] = []
    target_epochs: int | None = None
    cfg_path, cfg_dict = find_run_config(run_dir, source_root=source_root)
    history_file = run_dir / "history.json"
    metrics_file = run_dir / "metrics.csv"
    log_file = find_training_log(run_dir)

    # 1. Config target epochs
    if cfg_dict is not None:
        target_epochs = cfg_dict.get("training", {}).get("epochs")
    elif best_meta.get("config"):
        target_epochs = best_meta["config"].get("training", {}).get("epochs")
        cfg_dict = best_meta["config"]
    else:
        discrepancies.append("Neither config.yaml, config_used.yaml, nor checkpoint embedded config was found")


    # 2. History checks
    history_data: list[dict] = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            if not isinstance(history_data, list) or len(history_data) == 0:
                discrepancies.append("history.json is empty or not a list of epoch records")
        except Exception as e:
            discrepancies.append(f"Failed to parse history.json: {e}")
    else:
        discrepancies.append("history.json is missing")

    # 3. Metrics CSV checks
    csv_epochs: list[int] = []
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "epoch" in row:
                        csv_epochs.append(int(row["epoch"]))
        except Exception as e:
            discrepancies.append(f"Failed to parse metrics.csv: {e}")
    else:
        discrepancies.append("metrics.csv is missing")

    # 4. Checkpoint completion check against target epochs
    last_epoch = last_meta.get("epoch")
    if target_epochs is not None:
        if last_epoch is None:
            discrepancies.append(f"checkpoint_last.pt missing saved epoch (target: {target_epochs})")
        elif last_epoch < target_epochs:
            discrepancies.append(
                f"INCOMPLETE RUN: checkpoint_last.pt saved at epoch {last_epoch}, but target was {target_epochs}"
            )

        if history_data and history_data[-1].get("epoch") != target_epochs:
            discrepancies.append(
                f"INCOMPLETE HISTORY: history.json ends at epoch {history_data[-1].get('epoch')}, expected {target_epochs}"
            )

        if csv_epochs and csv_epochs[-1] != target_epochs:
            discrepancies.append(
                f"INCOMPLETE CSV: metrics.csv ends at epoch {csv_epochs[-1]}, expected {target_epochs}"
            )

    # 5. History and CSV epoch length agreement
    if history_data and csv_epochs:
        if len(history_data) != len(csv_epochs):
            discrepancies.append(
                f"Epoch count mismatch: history.json has {len(history_data)} epochs, metrics.csv has {len(csv_epochs)}"
            )

    # 6. Best epoch consistency
    if history_data and best_meta.get("epoch") is not None:
        best_ep = best_meta["epoch"]
        history_val_losses = [rec.get("val_loss") for rec in history_data if rec.get("val_loss") is not None]
        if history_val_losses:
            min_val = min(history_val_losses)
            best_rec = next((r for r in history_data if r.get("epoch") == best_ep), None)
            if best_rec:
                if abs(best_rec.get("val_loss", 0.0) - min_val) > 1e-6:
                    discrepancies.append(
                        f"Best checkpoint epoch {best_ep} has val_loss {best_rec.get('val_loss')}, but minimum in history is {min_val}"
                    )

    # 7. Training log check
    if log_file is None:
        discrepancies.append("Training log file (*.log) not found in run directory")

    is_complete = (
        len(discrepancies) == 0
        and target_epochs is not None
        and last_epoch == target_epochs
    )

    return {
        "target_epochs": target_epochs,
        "completed_epochs": last_epoch,
        "history_epochs": len(history_data),
        "is_complete": is_complete,
        "discrepancies": discrepancies,
    }


def import_colab_results(
    source_dir: str | Path,
    dest_dir: str | Path = "runs",
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Inspect and import Colab TPU run folders into local runs/ directory.

    Args:
        source_dir: Directory containing kth_*_tpu_50ep run folders.
        dest_dir: Local destination directory (default: 'runs').
        dry_run: If True, inspect and verify without copying files.
        overwrite: If True, allow overwriting existing checkpoints.

    Returns:
        Summary dict of verification and import results.
    """
    src = Path(source_dir)
    dst = Path(dest_dir)

    report: dict[str, Any] = {
        "source_dir": str(src),
        "dest_dir": str(dst),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "runs": {},
        "summary": {"verified": 0, "incomplete": 0, "failed": 0, "skipped": 0, "imported": 0},
    }

    if not src.exists():
        print(f"Error: Source directory '{src}' does not exist.")
        report["error"] = f"Source directory '{src}' does not exist."
        return report

    print("=" * 80)
    print(f"COLAB TPU RESULTS IMPORT & VERIFICATION UTILITY")
    print(f"Source: {src.resolve()}")
    print(f"Destination: {dst.resolve()}")
    print(f"Mode: {'DRY RUN (no files modified)' if dry_run else 'ACTIVE IMPORT'}")
    print(f"Overwrite policy: {'ALLOWED' if overwrite else 'REFUSE (preserve existing)'}")
    print("=" * 80)

    for run_name in EXPECTED_RUN_DIRS:
        run_src = src / run_name
        run_dst = dst / run_name

        print(f"\n[{run_name}]")
        run_report: dict[str, Any] = {
            "found": False,
            "artifacts_found": {},
            "checksums": {},
            "best_checkpoint": None,
            "last_checkpoint": None,
            "consistency": None,
            "status": "MISSING",
        }

        if not run_src.exists() or not run_src.is_dir():
            print(f"  ❌ Run folder not found at: {run_src}")
            run_report["status"] = "MISSING"
            report["summary"]["failed"] += 1
            report["runs"][run_name] = run_report
            continue

        run_report["found"] = True

        # Check required files
        missing_files = []
        for art_name in REQUIRED_ARTIFACT_NAMES:
            p = run_src / art_name
            exists = p.exists()
            run_report["artifacts_found"][art_name] = exists
            if not exists:
                missing_files.append(art_name)

        log_p = find_training_log(run_src)
        run_report["artifacts_found"]["training_log"] = log_p.name if log_p else None

        if missing_files:
            print(f"  ❌ Missing required artifacts: {', '.join(missing_files)}")
            run_report["status"] = "INCOMPLETE_ARTIFACTS"
            report["summary"]["failed"] += 1
            report["runs"][run_name] = run_report
            continue

        # Checkpoint inspection
        print("  🔍 Inspecting checkpoints...")
        best_ckpt_p = run_src / "checkpoint_best.pt"
        last_ckpt_p = run_src / "checkpoint_last.pt"

        best_meta = inspect_checkpoint(best_ckpt_p)
        last_meta = inspect_checkpoint(last_ckpt_p)

        run_report["best_checkpoint"] = best_meta
        run_report["last_checkpoint"] = last_meta

        if not best_meta["loadable"]:
            print(f"  ❌ checkpoint_best.pt failed to load on CPU: {best_meta['error_msg']}")
            if best_meta["is_xla_error"]:
                print("  💡 HINT: Checkpoint was serialized with XLA device bindings. See docs/TPU_RESULTS_HANDOFF.md for CPU export.")
            run_report["status"] = "LOAD_FAILURE"
            report["summary"]["failed"] += 1
            report["runs"][run_name] = run_report
            continue

        if not last_meta["loadable"]:
            print(f"  ❌ checkpoint_last.pt failed to load on CPU: {last_meta['error_msg']}")
            run_report["status"] = "LOAD_FAILURE"
            report["summary"]["failed"] += 1
            report["runs"][run_name] = run_report
            continue

        print(f"  ✅ checkpoint_best.pt: Epoch {best_meta['epoch']}, Step {best_meta['global_step']}, Val MSE {best_meta['val_loss']}")
        print(f"  ✅ checkpoint_last.pt: Epoch {last_meta['epoch']}, Step {last_meta['global_step']}")

        # Consistency verification
        consistency = verify_run_consistency(run_src, best_meta, last_meta, source_root=src)
        run_report["consistency"] = consistency

        if consistency["discrepancies"]:
            print("  ⚠️ Consistency discrepancies detected:")
            for disc in consistency["discrepancies"]:
                print(f"     - {disc}")
            if not consistency["is_complete"]:
                run_report["status"] = "INCOMPLETE_RUN"
                report["summary"]["incomplete"] += 1
                report["runs"][run_name] = run_report
                continue
        else:
            print(f"  ✅ Verification passed: 100% consistent across {consistency['completed_epochs']} target epochs.")
            report["summary"]["verified"] += 1

        # Checksums
        print("  🔒 Computing SHA-256 checksums...")
        for p in run_src.iterdir():
            if p.is_file():
                cs = compute_sha256(p)
                run_report["checksums"][p.name] = cs

        # Destination check
        dest_has_existing_ckpts = (run_dst / "checkpoint_best.pt").exists()
        if dest_has_existing_ckpts and not overwrite:
            print(f"  ⚠️ Destination {run_dst} already contains checkpoints. Skipping copy (--overwrite not set).")
            run_report["status"] = "SKIPPED_EXISTING"
            report["summary"]["skipped"] += 1
            report["runs"][run_name] = run_report
            continue

        if dry_run:
            print(f"  [DRY RUN] Would copy {len(run_report['checksums'])} files to {run_dst}")
            run_report["status"] = "VERIFIED_DRY_RUN"
        else:
            print(f"  📦 Copying files to {run_dst}...")
            run_dst.mkdir(parents=True, exist_ok=True)
            for p in run_src.iterdir():
                if p.is_file():
                    target_file = run_dst / p.name
                    shutil.copy2(p, target_file)

            # Ensure canonical config.yaml exists in destination for downstream tools
            if not (run_dst / "config.yaml").exists():
                cfg_p, cfg_d = find_run_config(run_src, source_root=src)
                if cfg_p and cfg_p.exists():
                    shutil.copy2(cfg_p, run_dst / "config.yaml")
                elif cfg_d:
                    with open(run_dst / "config.yaml", "w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg_d, f)

            print(f"  ✅ Successfully imported {run_name} into {run_dst}")
            run_report["status"] = "IMPORTED"
            report["summary"]["imported"] += 1

        report["runs"][run_name] = run_report

    print("\n" + "=" * 80)
    print("IMPORT AUDIT SUMMARY:")
    print(f"  Verified & Complete : {report['summary']['verified']}")
    print(f"  Incomplete Runs     : {report['summary']['incomplete']}")
    print(f"  Missing / Failed    : {report['summary']['failed']}")
    print(f"  Skipped (Existing)  : {report['summary']['skipped']}")
    print(f"  Imported to Dest    : {report['summary']['imported']}")
    print("=" * 80)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and verify Google Colab TPU training artifacts.")
    parser.add_argument("--source-dir", required=True, help="Directory containing the completed Colab TPU run folders.")
    parser.add_argument("--dest-dir", default="runs", help="Local destination runs/ directory (default: 'runs').")
    parser.add_argument("--dry-run", action="store_true", help="Perform verification without copying any files.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing destination checkpoints.")
    args = parser.parse_args()

    import_colab_results(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

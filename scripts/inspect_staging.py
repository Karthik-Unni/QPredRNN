import torch
import yaml
import json
from pathlib import Path

stage = Path(r"C:\temp\QPredRNN_Final_Integration\training_runs")
print("INSPECTING STAGING DIRECTORY:", stage)

for run_dir in sorted(stage.iterdir()):
    if not run_dir.is_dir():
        continue
    print("\n" + "=" * 60)
    print("STAGED RUN FOLDER:", run_dir.name)
    print("=" * 60)
    
    # 1. Config
    for cfg_name in ["config.yaml", "config_used.yaml"]:
        cfg_p = run_dir / cfg_name
        if cfg_p.exists():
            print(f"  Found {cfg_name}:")
            try:
                cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
                print(f"    model: {cfg.get('model', {}).get('name')}")
                print(f"    epochs: {cfg.get('training', {}).get('epochs')}")
                print(f"    batch_size: {cfg.get('training', {}).get('batch_size')}")
                print(f"    lr: {cfg.get('training', {}).get('lr')}")
            except Exception as e:
                print(f"    Error reading {cfg_name}: {e}")
                
    # 2. History
    hist_p = run_dir / "history.json"
    if hist_p.exists():
        try:
            hist = json.loads(hist_p.read_text(encoding="utf-8"))
            print(f"  history.json: {len(hist)} epochs (1 to {hist[-1]['epoch']})")
            min_entry = min(hist, key=lambda h: h["val_loss"])
            print(f"    Best val_loss: {min_entry['val_loss']:.6f} at epoch {min_entry['epoch']}")
            print(f"    Last val_loss: {hist[-1]['val_loss']:.6f} at epoch {hist[-1]['epoch']}")
        except Exception as e:
            print(f"  history.json error: {e}")
    else:
        print("  history.json: MISSING")
        
    # 3. Checkpoints
    for ckpt_name in ["checkpoint_best.pt", "checkpoint_last.pt"]:
        cp = run_dir / ckpt_name
        if cp.exists():
            try:
                c = torch.load(cp, map_location="cpu", weights_only=False)
                state = c.get("model_state_dict", {})
                print(f"  {ckpt_name} ({cp.stat().st_size:,} bytes):")
                print(f"    epoch: {c.get('epoch')}, global_step: {c.get('global_step')}")
                print(f"    val_loss: {c.get('val_loss')}, best_val_loss: {c.get('best_val_loss')}")
                print(f"    state_dict keys: {len(state)}")
                # Check for XLA or non-CPU tensors
                all_cpu = all(v.device.type == "cpu" for v in state.values() if isinstance(v, torch.Tensor))
                all_finite = all(torch.isfinite(v).all().item() for v in state.values() if isinstance(v, torch.Tensor))
                print(f"    all CPU tensors: {all_cpu}, all finite: {all_finite}")
            except Exception as e:
                print(f"  {ckpt_name} LOAD ERROR: {e}")
        else:
            print(f"  {ckpt_name}: MISSING")
            
    # 4. Logs
    logs = list(run_dir.glob("*.log"))
    print(f"  Log files: {[l.name for l in logs]}")

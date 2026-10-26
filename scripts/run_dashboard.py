"""Launch script for QPredRNN Research Dashboard."""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

def main():
    dashboard_path = Path(__file__).parents[1] / "qpred" / "dashboard.py"
    if not dashboard_path.exists():
        print(f"Error: Dashboard script not found at {dashboard_path}")
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)] + sys.argv[1:]
    print(f"[QPred] Launching Streamlit dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

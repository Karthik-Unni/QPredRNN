"""Launch QPredRNN Research Studio dashboard.

Always run this script using the GPU virtual environment:

    C:\\venvs\\qpred-gpu\\Scripts\\python.exe scripts\\run_dashboard.py

Or launch streamlit directly:

    C:\\venvs\\qpred-gpu\\Scripts\\python.exe -m streamlit run qpred/dashboard.py
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

# Warn if run with the wrong interpreter
_expected_venv = Path(r"C:\venvs\qpred-gpu")
_running_exe = Path(sys.executable).resolve()
if _expected_venv.exists() and _expected_venv not in _running_exe.parents:
    print(
        f"[QPred] ⚠️  WARNING: Running with {_running_exe}\n"
        f"[QPred]    Expected venv: {_expected_venv / 'Scripts' / 'python.exe'}\n"
        f"[QPred]    To use GPU/PennyLane, run:\n"
        f"[QPred]      {_expected_venv / 'Scripts' / 'python.exe'} scripts/run_dashboard.py"
    )


def main() -> None:
    dashboard_path = Path(__file__).parents[1] / "qpred" / "dashboard.py"
    if not dashboard_path.exists():
        print(f"[QPred] ❌ Dashboard script not found at {dashboard_path}")
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
           "--server.headless=false"] + sys.argv[1:]
    print(f"[QPred] 🚀 Launching: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

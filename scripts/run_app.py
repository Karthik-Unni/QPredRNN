"""Launch the QPredRNN Research Dashboard (React + FastAPI).

Usage:
  C:\\venvs\\qpred-gpu\\Scripts\\python.exe scripts\\run_app.py

Options:
  --port PORT       Server port (default: 8000)
  --host HOST       Server host (default: 127.0.0.1)
  --reload          Enable auto-reload on python file changes
  --dev             Also start the Vite frontend development server
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).parents[1].resolve()
EXPECTED_VENV = Path(r"C:\venvs\qpred-gpu")
RUNNING_EXE = Path(sys.executable).resolve()

if EXPECTED_VENV.exists() and EXPECTED_VENV not in RUNNING_EXE.parents:
    print(
        f"[QPred] [!] WARNING: Running with {RUNNING_EXE}\n"
        f"[QPred]     Expected GPU venv: {EXPECTED_VENV / 'Scripts' / 'python.exe'}\n"
        f"[QPred]     Recommendation: run with {EXPECTED_VENV / 'Scripts' / 'python.exe'} scripts/run_app.py\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QPredRNN React + FastAPI Studio Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    parser.add_argument("--dev", action="store_true", help="Launch Vite dev server concurrently")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    # Ensure frontend build exists if not in dev mode
    dist_dir = ROOT / "frontend" / "dist"
    if not dist_dir.exists() and not args.dev:
        print("[QPred] [*] Building React frontend first...")
        frontend_dir = ROOT / "frontend"
        try:
            subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True, shell=True)
            print("[QPred] [OK] Frontend build complete.")
        except Exception as e:
            print(f"[QPred] [!] Frontend build warning: {e}")

    url = f"http://{args.host}:{args.port}"
    print("\n========================================================")
    print("  QPredRNN Quantum Spatiotemporal Research Studio")
    print(f"  Serving at: {url}")
    print(f"  API Docs at: {url}/docs")
    print("========================================================\n")

    if not args.no_browser:
        try:
            import threading
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    import uvicorn
    uvicorn.run(
        "qpred.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()

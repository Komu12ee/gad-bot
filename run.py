#!/usr/bin/env python3
"""
GAD-BOT Runner: Launches both the Flask Backend and Node.js Web UI for GAD.

- Flask API Backend: http://127.0.0.1:5003
- Node.js UI & Proxy: http://localhost:3003
- Temporal Graph & History Explorer: http://localhost:3003/graph.html
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# Set system env vars
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PIPELINE"] = os.getenv("PIPELINE", "fg2")
os.environ["FLASK_PORT"] = os.getenv("FLASK_PORT", "5003")
os.environ["PORT"] = os.getenv("PORT", "3003")
os.environ["NODE_ENV"] = os.getenv("NODE_ENV", "development")

FLASK_PORT = os.environ["FLASK_PORT"]
NODE_PORT = os.environ["PORT"]

CORE_DIR = ROOT_DIR / "core"
NODE_DIR = CORE_DIR / "05_webui" / "nodejs"
FLASK_APP = CORE_DIR / "05_webui" / "app.py"


def print_banner():
    print("\n" + "═" * 70)
    print("  🏛️  GAD-BOT: Governance Assistant & Temporal Knowledge Graph")
    print("  Government of Chhattisgarh · General Administration Department")
    print("═" * 70)
    print(f"  🌐  Web Interface     →  http://localhost:{NODE_PORT}")
    print(f"  📊  Temporal Graph    →  http://localhost:{NODE_PORT}/graph.html")
    print(f"  ⚡  Flask API Backend →  http://127.0.0.1:{FLASK_PORT}")
    print(f"  🤖  LLM Provider      →  {os.getenv('LLM_BASE_URL', 'http://103.51.8.88/v1')}")
    print(f"  🎯  LLM Model         →  {os.getenv('LLM_MODEL', 'ocr-32b')}")
    print(f"  📚  Temporal Graph DB →  data/fg2/temporal_graph.db (5,305 nodes, 10,630 edges)")
    print("═" * 70)
    print("  Press Ctrl+C to stop all services.\n")


def check_graph_db():
    graph_db = ROOT_DIR / "data" / "fg2" / "temporal_graph.db"
    if not graph_db.exists():
        print("⚠️ Temporal graph database not found. Building now...")
        build_script = CORE_DIR / "04_embeddings_and_kg" / "scripts" / "build_graph.py"
        subprocess.run([sys.executable, str(build_script)], check=True)
    else:
        print("✅ Temporal graph database ready.")


def auto_update_docs():
    docs_script = ROOT_DIR / "scripts" / "update_docs.py"
    if docs_script.exists():
        try:
            subprocess.run([sys.executable, str(docs_script)], check=True, stdout=subprocess.DEVNULL)
            print("📄 Automatically updated system documentation (DOCUMENTATION.md).")
        except Exception as e:
            print(f"⚠️ Could not auto-update documentation: {e}")


def main():
    print_banner()
    check_graph_db()
    auto_update_docs()

    env = os.environ.copy()

    # 1. Start Flask API process
    print(f"🚀 Starting Flask backend on port {FLASK_PORT}...")
    flask_proc = subprocess.Popen(
        [sys.executable, str(FLASK_APP)],
        cwd=str(ROOT_DIR),
        env=env
    )

    # Allow Flask a moment to initialize
    time.sleep(1.5)

    # 2. Start Node.js Web Server process
    print(f"🚀 Starting Node.js UI server on port {NODE_PORT}...")
    node_server = NODE_DIR / "server.js"
    node_cmd = ["node", str(node_server)]
    
    node_proc = subprocess.Popen(
        node_cmd,
        cwd=str(NODE_DIR),
        env=env
    )

    print("\n✨ Both GAD-BOT services are running!\n")

    def handle_exit(signum, frame):
        print("\n🛑 Shutting down GAD-BOT services...")
        try:
            node_proc.terminate()
        except Exception:
            pass
        try:
            flask_proc.terminate()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        while True:
            time.sleep(1)
            # If any process dies unexpectedly, alert
            if flask_proc.poll() is not None:
                print(f"⚠️ Flask backend process stopped with code {flask_proc.returncode}")
                break
            if node_proc.poll() is not None:
                print(f"⚠️ Node.js UI server process stopped with code {node_proc.returncode}")
                break
    except KeyboardInterrupt:
        handle_exit(None, None)


if __name__ == "__main__":
    main()

"""The answer model, switched from the chat page (admins: "Answer model" in the sidebar).

One model on the GPU VM's :8000 answers all three bots. A switch runs
`bash deploy/gpu-vm.sh chat-model <key>` as its own process (it outlives this app):
the model server stops, the new one loads — a minute or two without answers — and
the app keeps running; the answer client notices the new model by itself
(llm_client re-reads /v1/models). The choices are deploy/chat-models.conf; progress
is the script's $STATE_DIR/chat-switch.json, which this module reads.

Only where this machine runs the model server itself (the GPU VM: gpu-vm.sh has
served a chat model, LLM_PROVIDER=gpu on localhost:8000); elsewhere the panel says so.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Blueprint, jsonify, request

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / "deploy" / "chat-models.conf"
SCRIPT = REPO / "deploy" / "gpu-vm.sh"
STATE_DIR = REPO / ".storage" / "state" / "gpu-vm"
LOG = REPO / "logs" / "gpu-vm" / "chat-switch.log"
BASH = os.getenv("BASH_BIN") or "bash"
SWITCH_MINUTES = 15          # a "switching" status older than this, with its process gone, is stale

_lock = threading.Lock()
_proc = None                 # the switch this app started (polled, so it is reaped when it ends)


def choices() -> list[dict]:
    """[{key, label, model, flags}] from deploy/chat-models.conf."""
    out = []
    try:
        lines = CONF.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0] and parts[2]:
            out.append({"key": parts[0], "label": parts[1] or parts[0], "model": parts[2],
                        "flags": parts[3] if len(parts) > 3 else ""})
    return out


def _base_url() -> str:
    url = (os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
    if url.endswith("/models"):
        url = url[:-7]
    return url


def managed() -> bool:
    """This machine runs the answer model itself, the way gpu-vm.sh does."""
    if os.getenv("LLM_MANAGE_SERVER", "1").strip() == "0":
        return False
    if not SCRIPT.exists() or not (STATE_DIR / "chat.spec").exists():
        return False
    if (os.getenv("LLM_PROVIDER") or "").strip().lower() != "gpu":
        return False
    return _base_url().rstrip("/").removesuffix("/v1") in ("http://127.0.0.1:8000", "http://localhost:8000")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def served_model() -> str | None:
    """The model id :8000 serves now, or None while it is down."""
    try:
        r = requests.get(f"{_base_url()}/models", timeout=3)
        ids = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict)]
        return ids[0] if r.status_code == 200 and ids else None
    except (requests.RequestException, ValueError):
        return None


def _pid_alive(pid) -> bool:
    if _proc is not None and _proc.poll() is not None and _proc.pid == pid:
        return False                       # ours, and it has ended (poll() reaps it)
    if os.name == "nt":                    # os.kill(pid, 0) TERMINATES a process on Windows
        return False                       # (and no switch runs there)
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def switch_status() -> dict | None:
    """The last switch as the script recorded it; a 'switching' left by a process
    that is gone (killed, machine restarted) reads as failed."""
    try:
        st = json.loads((STATE_DIR / "chat-switch.json").read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    if st.get("state") == "switching" and not _pid_alive(st.get("pid")):
        try:
            age = time.time() - datetime.fromisoformat(st["at"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError):
            age = SWITCH_MINUTES * 60 + 1
        if age > 30:      # the script writes its own status within a second of starting
            st = {**st, "state": "failed", "message": "The switch stopped before it finished — see logs/gpu-vm/chat-switch.log."}
    return st


def status() -> dict:
    opts = choices()
    served = served_model()
    by_model = {c["model"]: c for c in opts}
    cur = by_model.get(served) if served else None
    st = switch_status()
    return {
        "available": managed(),
        "choices": [{"key": c["key"], "label": c["label"], "model": c["model"]} for c in opts],
        "served": served, "current": cur["key"] if cur else None,
        "current_label": cur["label"] if cur else served,
        "switch": st,
        "switching": bool(st and st.get("state") == "switching"),
        "ocr_running": _port_open(8001) or _port_open(8002),
    }


def start_switch(key: str, user: str | None = None) -> dict:
    """Start `gpu-vm.sh chat-model <key>` in its own session. Raises ValueError with
    a message for the page when it cannot or need not run."""
    global _proc
    with _lock:
        target = next((c for c in choices() if c["key"] == key), None)
        if not target:
            raise ValueError("Unknown model.")
        if not managed():
            raise ValueError("This machine does not run the answer model itself, so it cannot switch it.")
        st = switch_status()
        if st and st.get("state") == "switching":
            raise ValueError("A switch is already running.")
        if _port_open(8001) or _port_open(8002):
            raise ValueError("The OCR is using the GPU — no answer model runs beside it. Stop the OCR first.")
        if served_model() == target["model"]:
            raise ValueError(f"{target['label']} is already answering.")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG, "a", encoding="utf-8")
        log.write(f"\n=== {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ} switch to {key} by {user or '?'}\n")
        log.flush()
        env = {**os.environ, "GPU_VM_NO_TMUX": "1"}
        for k in ("CHAT_MODEL", "CHAT_VLLM_ARGS"):          # the choice decides, not this app's env
            env.pop(k, None)
        # "switching" at once for the page — written BEFORE the script starts, so a quick
        # refusal it writes (another run active, OCR on the GPU) is never overwritten.
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "chat-switch.json").write_text(json.dumps({
            "state": "switching", "key": key, "message": f"Loading {target['label']}…",
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "pid": None}),
            encoding="utf-8")
        try:
            proc = _proc = subprocess.Popen([BASH, str(SCRIPT), "chat-model", key], cwd=str(REPO),
                                            env=env, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            (STATE_DIR / "chat-switch.json").write_text(json.dumps({
                "state": "failed", "key": key, "message": f"The switch could not start: {e}",
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "pid": None}),
                encoding="utf-8")
            raise ValueError(f"The switch could not start: {e}") from e
        finally:
            log.close()
        print(f"[chat-model] switch to {key} ({target['model']}) by {user or '?'} — pid {proc.pid}", flush=True)
        return {"key": key, "label": target["label"], "pid": proc.pid}


def friendly_error(err) -> str | None:
    """While a switch runs, an answer that failed for want of a model server gets a
    plain explanation instead of a connection error."""
    st = switch_status()
    if st and st.get("state") == "switching":
        label = next((c["label"] for c in choices() if c["key"] == st.get("key")), "the new model")
        return (f"The answer model is being switched to {label} — the bots answer again in a "
                f"minute or two. Please ask again then.")
    return None


def blueprint() -> Blueprint:
    """/api/admin/chat-model — admins only (the Node server gates /api/admin)."""
    bp = Blueprint("chat_model", __name__)

    @bp.route("/api/admin/chat-model", methods=["GET"])
    def chat_model_get():
        return jsonify({"success": True, **status()})

    @bp.route("/api/admin/chat-model", methods=["POST"])
    def chat_model_post():
        data = request.get_json(silent=True) or {}
        try:
            started = start_switch(str(data.get("key") or ""), request.headers.get("X-Gad-User") or None)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 409
        return jsonify({"success": True, "started": started, **status()}), 202

    return bp

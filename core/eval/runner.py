"""Asynchronous evaluation execution runner with cancellation and progress reporting."""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_CORE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CORE_DIR.parent
for _p in [str(_REPO_ROOT), str(_CORE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from core.eval.registry import REGISTRY
    from core.eval.storage import EvalStorage
except ModuleNotFoundError:
    from eval.registry import REGISTRY
    from eval.storage import EvalStorage


class EvalRunner:
    def __init__(self, storage: EvalStorage):
        self.storage = storage
        self._lock = threading.Lock()
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

    def start_job(
        self,
        component_id: str,
        batch_path: str,
        dataset_name: str,
        config: Optional[Dict[str, Any]] = None,
        evaluate_fn: Optional[Callable[..., Any]] = None,
    ) -> str:
        """Start an evaluation run in a background daemon thread."""
        comp_def = REGISTRY.get(component_id)
        if not comp_def:
            raise ValueError(f"Unknown component: {component_id}")

        run_id = f"eval_{component_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        cfg = config or {}

        # Register run in SQLite storage
        self.storage.create_run(
            run_id=run_id,
            component=component_id,
            dataset_name=dataset_name,
            model_version=str(cfg.get("model_version", "v1.0")),
            config=cfg,
            n_rows=0,
        )

        cancel_event = threading.Event()
        job_info = {
            "run_id": run_id,
            "component": component_id,
            "status": "running",
            "progress": 0,
            "total": 0,
            "message": "Initializing...",
            "cancel_event": cancel_event,
            "thread": None,
            "error": None,
        }

        with self._lock:
            self._active_jobs[run_id] = job_info

        def _worker():
            try:
                def _on_progress(cur, tot, msg):
                    with self._lock:
                        job_info["progress"] = cur
                        job_info["total"] = tot
                        job_info["message"] = msg

                fn = evaluate_fn
                if fn is None:
                    if component_id == "hybrid_retrieval":
                        try:
                            from core.eval.components.hybrid_retrieval import evaluate_hybrid_retrieval
                        except ModuleNotFoundError:
                            from eval.components.hybrid_retrieval import evaluate_hybrid_retrieval
                        fn = evaluate_hybrid_retrieval
                    else:
                        raise NotImplementedError(f"Evaluator for {component_id} is not yet registered.")

                # Execute component evaluator
                eval_res = fn(
                    batch_path=batch_path,
                    config=cfg,
                    storage=self.storage,
                    run_id=run_id,
                    progress_callback=_on_progress,
                )

                if cancel_event.is_set():
                    self.storage.finish_run(
                        run_id=run_id,
                        status="stopped",
                        n_errors=eval_res.get("n_errors", 0),
                        kpis=eval_res.get("kpis", {}),
                        slices=eval_res.get("slices", {}),
                        latency=eval_res.get("latency_ms", {}),
                    )
                    with self._lock:
                        job_info["status"] = "stopped"
                        job_info["message"] = "Run was cancelled by user."
                    return

                # Record completion
                self.storage.finish_run(
                    run_id=run_id,
                    status="done",
                    n_errors=eval_res.get("n_errors", 0),
                    kpis=eval_res.get("kpis", {}),
                    slices=eval_res.get("slices", {}),
                    latency=eval_res.get("latency_ms", {}),
                )
                with self._lock:
                    job_info["status"] = "done"
                    job_info["message"] = f"Completed {eval_res.get('n_rows', 0)} rows."

            except Exception as e:
                err_msg = str(e)
                print(f"[eval_runner] Run {run_id} failed: {err_msg}")
                self.storage.finish_run(
                    run_id=run_id,
                    status="failed",
                    n_errors=1,
                    kpis={},
                    slices={},
                    latency={},
                )
                with self._lock:
                    job_info["status"] = "failed"
                    job_info["error"] = err_msg
                    job_info["message"] = f"Error: {err_msg}"

        th = threading.Thread(target=_worker, name=f"eval-{run_id}", daemon=True)
        job_info["thread"] = th
        th.start()

        return run_id

    def cancel_job(self, run_id: str) -> bool:
        with self._lock:
            job = self._active_jobs.get(run_id)
            if job and job["status"] == "running":
                job["cancel_event"].set()
                job["status"] = "stopping"
                return True
        return False

    def get_status(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._active_jobs.get(run_id)
            if job:
                return {
                    "run_id": run_id,
                    "status": job["status"],
                    "progress": job["progress"],
                    "total": job["total"],
                    "message": job["message"],
                    "error": job["error"],
                }
        # Fallback to persistent DB
        stored = self.storage.get_run(run_id)
        if stored:
            return {
                "run_id": run_id,
                "status": stored["status"],
                "progress": stored["n_rows"],
                "total": stored["n_rows"],
                "message": f"Run {stored['status']}",
                "error": None,
            }
        return {"run_id": run_id, "status": "not_found"}


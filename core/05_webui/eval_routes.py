"""Flask REST API blueprint for Component Evaluation.

Exposes:
- POST /eval/component/<component>/upload
- POST /eval/component/<component>/run
- GET  /eval/component/<component>/runs/<run_id>
- GET  /eval/component/<component>/template
- GET  /eval/runs
- POST /eval/runs/<run_id>/cancel
- GET  /eval/runs/<run_id>/export
(Also mirrored under /api/eval/* for frontends hitting the /api proxy)
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # = core/
_REPO_ROOT = _PROJECT_ROOT.parent   # = repo root
for _p in [str(_REPO_ROOT), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Blueprint, Response, jsonify, request, send_file
import pipeline_config as pc

try:
    from core.eval.registry import REGISTRY
    from core.eval.runner import EvalRunner
    from core.eval.storage import EvalStorage
    from core.eval.validator import parse_and_validate_batch
except ModuleNotFoundError:
    from eval.registry import REGISTRY
    from eval.runner import EvalRunner
    from eval.storage import EvalStorage
    from eval.validator import parse_and_validate_batch

# Singleton storage and runner
_STORAGE = EvalStorage(pc.DATA_ROOT)
_RUNNER = EvalRunner(_STORAGE)


def create_eval_blueprint() -> Blueprint:
    bp = Blueprint("eval_routes", __name__)

    # ── Upload validation ────────────────────────────────────────────────────
    def _handle_upload(component: str):
        if component not in REGISTRY:
            return jsonify({"success": False, "error": f"Unknown component '{component}'"}), 404

        file = request.files.get("file")
        if not file:
            # Check for JSON raw body
            body_json = request.get_json(silent=True)
            if body_json and "rows" in body_json:
                content_bytes = json.dumps(body_json).encode("utf-8")
                filename = "upload.json"
            else:
                return jsonify({"success": False, "error": "No file uploaded (field 'file')"}), 400
        else:
            filename = file.filename or "upload.csv"
            content_bytes = file.read()

        is_valid, rows, errors = parse_and_validate_batch(content_bytes, filename, component)

        # Save valid upload to a persistent staging directory for later run execution
        upload_id = f"staged_{component}_{len(rows)}"
        staged_dir = pc.DATA_ROOT / "eval" / "staged"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_file = staged_dir / f"{upload_id}_{Path(filename).name}"
        staged_file.write_bytes(content_bytes)

        return jsonify({
            "success": True,
            "valid": is_valid,
            "filename": filename,
            "staged_path": str(staged_file),
            "row_count": len(rows),
            "error_count": len(errors),
            "errors": errors[:50],  # first 50 error rows
            "sample_row": rows[0] if rows else None,
        })

    @bp.route("/eval/component/<component>/upload", methods=["POST"])
    @bp.route("/api/eval/component/<component>/upload", methods=["POST"])
    def upload_route(component):
        return _handle_upload(component)

    # ── Run trigger ──────────────────────────────────────────────────────────
    def _handle_run(component: str):
        if component not in REGISTRY:
            return jsonify({"success": False, "error": f"Unknown component '{component}'"}), 404

        data = request.get_json(silent=True) or {}
        staged_path = data.get("staged_path")
        dataset_name = data.get("dataset_name") or "dataset.csv"
        cfg = data.get("config") or {}

        # Allow uploading directly during run if staged_path not provided
        if not staged_path or not Path(staged_path).exists():
            return jsonify({"success": False, "error": f"Invalid or missing 'staged_path': {staged_path}"}), 400

        try:
            run_id = _RUNNER.start_job(
                component_id=component,
                batch_path=staged_path,
                dataset_name=dataset_name,
                config=cfg,
            )
            return jsonify({"success": True, "run_id": run_id, "status": "running"}), 201
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @bp.route("/eval/component/<component>/run", methods=["POST"])
    @bp.route("/api/eval/component/<component>/run", methods=["POST"])
    def run_route(component):
        return _handle_run(component)

    # ── Run details & polling ────────────────────────────────────────────────
    def _handle_get_run(component: str, run_id: str):
        job_status = _RUNNER.get_status(run_id)
        run_record = _STORAGE.get_run(run_id)

        if not run_record and job_status.get("status") == "not_found":
            return jsonify({"success": False, "error": "Run not found"}), 404

        # Compare with previous run for delta arrows
        prev_run = _STORAGE.get_previous_run(component, before_run_id=run_id)
        failures = _STORAGE.get_run_rows(run_id, failed_only=True, limit=200)

        response = {
            "success": True,
            "run_id": run_id,
            "component": component,
            "status": job_status.get("status", run_record.get("status") if run_record else "unknown"),
            "progress": job_status.get("progress", run_record.get("n_rows") if run_record else 0),
            "total": job_status.get("total", run_record.get("n_rows") if run_record else 0),
            "message": job_status.get("message", ""),
            "run": run_record,
            "failures": failures,
            "previous_run_kpis": prev_run.get("kpis") if prev_run else None,
        }
        return jsonify(response)

    @bp.route("/eval/component/<component>/runs/<run_id>", methods=["GET"])
    @bp.route("/api/eval/component/<component>/runs/<run_id>", methods=["GET"])
    def get_run_route(component, run_id):
        return _handle_get_run(component, run_id)

    # ── Template download ────────────────────────────────────────────────────
    def _handle_template(component: str):
        if component not in REGISTRY:
            return jsonify({"success": False, "error": f"Unknown component '{component}'"}), 404

        sample_file = Path(__file__).resolve().parent.parent / "eval" / "samples" / f"sample_{component}.csv"
        if not sample_file.exists():
            # Dynamically build a basic template from registered schema
            comp_def = REGISTRY[component]
            cols = [c.name for c in comp_def.columns]
            content = ",".join(cols) + "\n"
            return Response(
                "\ufeff" + content,
                mimetype="text/csv",
                headers={"Content-Disposition": f"attachment; filename=template_{component}.csv"},
            )

        return send_file(
            str(sample_file),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"sample_{component}.csv",
        )

    @bp.route("/eval/component/<component>/template", methods=["GET"])
    @bp.route("/api/eval/component/<component>/template", methods=["GET"])
    def template_route(component):
        return _handle_template(component)

    # ── Run history ──────────────────────────────────────────────────────────
    def _handle_list_runs():
        comp = request.args.get("component")
        limit = int(request.args.get("limit", "50"))
        runs = _STORAGE.list_runs(component=comp, limit=limit)
        return jsonify({"success": True, "runs": runs})

    @bp.route("/eval/runs", methods=["GET"])
    @bp.route("/api/eval/runs", methods=["GET"])
    def list_runs_route():
        return _handle_list_runs()

    # ── Run cancellation ─────────────────────────────────────────────────────
    @bp.route("/eval/runs/<run_id>/cancel", methods=["POST"])
    @bp.route("/api/eval/runs/<run_id>/cancel", methods=["POST"])
    def cancel_run_route(run_id):
        cancelled = _RUNNER.cancel_job(run_id)
        return jsonify({"success": True, "cancelled": cancelled})

    # ── CSV export of row results ────────────────────────────────────────────
    @bp.route("/eval/runs/<run_id>/export", methods=["GET"])
    @bp.route("/api/eval/runs/<run_id>/export", methods=["GET"])
    def export_run_route(run_id):
        csv_data = _STORAGE.export_csv(run_id)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=eval-run-{run_id}.csv"},
        )

    # ── Components metadata list ─────────────────────────────────────────────
    @bp.route("/eval/components", methods=["GET"])
    @bp.route("/api/eval/components", methods=["GET"])
    def list_components_route():
        data = {}
        for cid, cdef in REGISTRY.items():
            data[cid] = {
                "id": cdef.id,
                "label": cdef.label,
                "description": cdef.description,
                "columns": [{"name": c.name, "required": c.required, "type": c.col_type, "desc": c.description} for c in cdef.columns],
                "kpis": {k: {"target": kd.target, "direction": kd.direction, "formula": kd.formula, "desc": kd.description, "unit": kd.unit} for k, kd in cdef.kpis.items()},
            }
        return jsonify({"success": True, "components": data})

    return bp


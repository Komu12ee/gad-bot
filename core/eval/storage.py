"""SQLite-backed persistent evaluation run storage.

Stores run summaries, per-row outputs, config snapshots, and failures.
Supports delta comparisons against previous runs and CSV exports.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_LOCAL = threading.local()


def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = getattr(_LOCAL, f"conn_{db_path}", None)
    if conn is None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        setattr(_LOCAL, f"conn_{db_path}", conn)
    return conn


class EvalStorage:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.db_path = self.data_root / "eval" / "eval_runs.db"
        self._init_db()

    def _init_db(self):
        conn = _get_connection(self.db_path)
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    dataset_name TEXT,
                    model_version TEXT,
                    config_json TEXT,
                    status TEXT NOT NULL,
                    n_rows INTEGER DEFAULT 0,
                    n_errors INTEGER DEFAULT 0,
                    kpis_json TEXT,
                    slices_json TEXT,
                    latency_json TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_eval_runs_comp ON eval_runs(component, created_at DESC);

                CREATE TABLE IF NOT EXISTS eval_run_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    row_id TEXT NOT NULL,
                    input_json TEXT,
                    expected_json TEXT,
                    actual_json TEXT,
                    passed INTEGER NOT NULL DEFAULT 1,
                    reason TEXT,
                    latency_ms REAL DEFAULT 0.0,
                    slice_tags_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_eval_rows_run ON eval_run_rows(run_id, passed);
            """)

    def create_run(
        self,
        run_id: str,
        component: str,
        dataset_name: str,
        model_version: str = "",
        config: Optional[Dict[str, Any]] = None,
        n_rows: int = 0,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_connection(self.db_path)
        with conn:
            conn.execute(
                """
                INSERT INTO eval_runs (
                    run_id, component, dataset_name, model_version, config_json,
                    status, n_rows, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    component,
                    dataset_name,
                    model_version,
                    json.dumps(config or {}, ensure_ascii=False),
                    "running",
                    n_rows,
                    now,
                ),
            )
        return self.get_run(run_id) or {}

    def finish_run(
        self,
        run_id: str,
        status: str,
        n_errors: int,
        kpis: Dict[str, Any],
        slices: Dict[str, Any],
        latency: Dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_connection(self.db_path)
        with conn:
            conn.execute(
                """
                UPDATE eval_runs SET
                    status = ?,
                    n_errors = ?,
                    kpis_json = ?,
                    slices_json = ?,
                    latency_json = ?,
                    finished_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    n_errors,
                    json.dumps(kpis, ensure_ascii=False),
                    json.dumps(slices, ensure_ascii=False),
                    json.dumps(latency, ensure_ascii=False),
                    now,
                    run_id,
                ),
            )

    def record_row(
        self,
        run_id: str,
        row_id: str,
        input_data: Any,
        expected_data: Any,
        actual_data: Any,
        passed: bool,
        reason: str = "",
        latency_ms: float = 0.0,
        slice_tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = _get_connection(self.db_path)
        with conn:
            conn.execute(
                """
                INSERT INTO eval_run_rows (
                    run_id, row_id, input_json, expected_json, actual_json,
                    passed, reason, latency_ms, slice_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(row_id),
                    json.dumps(input_data, ensure_ascii=False, default=str),
                    json.dumps(expected_data, ensure_ascii=False, default=str),
                    json.dumps(actual_data, ensure_ascii=False, default=str),
                    1 if passed else 0,
                    reason,
                    latency_ms,
                    json.dumps(slice_tags or {}, ensure_ascii=False),
                ),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = _get_connection(self.db_path)
        cur = conn.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["config"] = json.loads(res.pop("config_json") or "{}")
        res["kpis"] = json.loads(res.pop("kpis_json") or "{}")
        res["slices"] = json.loads(res.pop("slices_json") or "{}")
        res["latency"] = json.loads(res.pop("latency_json") or "{}")
        return res

    def list_runs(self, component: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        conn = _get_connection(self.db_path)
        query = "SELECT * FROM eval_runs"
        params: List[Any] = []
        if component:
            query += " WHERE component = ?"
            params.append(component)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cur = conn.execute(query, params)
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json") or "{}")
            d["kpis"] = json.loads(d.pop("kpis_json") or "{}")
            d["slices"] = json.loads(d.pop("slices_json") or "{}")
            d["latency"] = json.loads(d.pop("latency_json") or "{}")
            out.append(d)
        return out

    def get_previous_run(self, component: str, before_run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        conn = _get_connection(self.db_path)
        if before_run_id:
            cur = conn.execute(
                """
                SELECT * FROM eval_runs
                WHERE component = ? AND status = 'done' AND created_at < (
                    SELECT created_at FROM eval_runs WHERE run_id = ?
                )
                ORDER BY created_at DESC LIMIT 1
                """,
                (component, before_run_id),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM eval_runs
                WHERE component = ? AND status = 'done'
                ORDER BY created_at DESC LIMIT 1
                """,
                (component,),
            )
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["kpis"] = json.loads(res.pop("kpis_json") or "{}")
        return res

    def get_run_rows(self, run_id: str, failed_only: bool = False, limit: int = 1000) -> List[Dict[str, Any]]:
        conn = _get_connection(self.db_path)
        query = "SELECT * FROM eval_run_rows WHERE run_id = ?"
        params: List[Any] = [run_id]
        if failed_only:
            query += " AND passed = 0"
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)

        cur = conn.execute(query, params)
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["input"] = json.loads(d.pop("input_json") or "{}")
            d["expected"] = json.loads(d.pop("expected_json") or "{}")
            d["actual"] = json.loads(d.pop("actual_json") or "{}")
            d["slice_tags"] = json.loads(d.pop("slice_tags_json") or "{}")
            d["passed"] = bool(d["passed"])
            rows.append(d)
        return rows

    def export_csv(self, run_id: str) -> str:
        """Export row-level results to CSV format with UTF-8 BOM for Excel support."""
        rows = self.get_run_rows(run_id, limit=10000)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Row ID", "Passed", "Reason", "Latency (ms)", "Input", "Expected", "Actual", "Slice Tags"])
        for r in rows:
            w.writerow([
                r["row_id"],
                "PASS" if r["passed"] else "FAIL",
                r["reason"],
                r["latency_ms"],
                json.dumps(r["input"], ensure_ascii=False),
                json.dumps(r["expected"], ensure_ascii=False),
                json.dumps(r["actual"], ensure_ascii=False),
                json.dumps(r["slice_tags"], ensure_ascii=False),
            ])
        return "\ufeff" + buf.getvalue()


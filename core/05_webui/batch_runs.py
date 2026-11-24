"""Batch test runs (admins): a list of questions answered by the bot one after
another, kept on disk and downloadable as CSV (opens in Excel) or JSON — the Batch
test panel (nodejs/public/batch-test.html).

Answers come from the same code as the chat (app.py answer_one), each question on
its own, without chat history. One run at a time and one question at a time, so a
test takes no more of the answer model than one more person chatting. A run goes on
when the browser is closed; an app restart stops it ("stopped"), and "Answer the
rest" picks up where it ended — that also retries the questions that failed.

Runs live in <DATA_ROOT>/batch_tests/<id>.json.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

MAX_QUESTIONS = int(os.getenv("BATCH_MAX_QUESTIONS", "500"))
MAX_CHARS = 2000                 # one question, or one expected answer
MODES = ("normal", "thinking")
STOP_AFTER_ERRORS = 5            # failures in a row: the bot is down — stop, don't fail every question
_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Runs:
    def __init__(self, root: Path, answer):
        """`answer(question, mode)` -> {"answer", "sources": [{"label", "url"}],
        "retrieved": [labels], "model"}; raises when the bot could not answer (the
        message is kept as the question's error)."""
        self.root = Path(root)
        self.answer = answer
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._current: str | None = None
        self._stop: set[str] = set()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._summaries: dict[str, tuple[float, dict]] = {}
        self._recover()

    # ── files ────────────────────────────────────────────────────────────────
    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    def load(self, rid: str) -> dict | None:
        if not isinstance(rid, str) or not _ID.fullmatch(rid):
            return None
        try:
            return json.loads(self._path(rid).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, run: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f"{run['id']}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
        for attempt in range(40):
            try:
                os.replace(tmp, self._path(run["id"]))
                return
            except PermissionError:      # Windows: the file is open in a reader at this moment
                if attempt == 39:
                    raise
                time.sleep(0.05)

    def _recover(self) -> None:
        """A run that was queued or answering when the app stopped is "stopped"."""
        for p in sorted(self.root.glob("*.json")) if self.root.is_dir() else []:
            run = self.load(p.stem)
            if run and run.get("status") in ("queued", "running"):
                run.update(status="stopped", stop_reason="the app restarted")
                self.save(run)

    @staticmethod
    def summary(run: dict) -> dict:
        items = run.get("items") or []
        done = [i for i in items if i.get("status") == "done"]
        secs = [i["seconds"] for i in done if i.get("seconds") is not None]
        return {k: run.get(k) for k in ("id", "name", "mode", "created_at", "started_by", "status",
                                        "started_at", "finished_at", "stop_reason")} | {
            "total": len(items), "answered": len(done),
            "errors": sum(1 for i in items if i.get("status") == "error"),
            "models": sorted({i["model"] for i in items if i.get("model")}),
            "avg_seconds": round(sum(secs) / len(secs), 1) if secs else None}

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json"), reverse=True) if self.root.is_dir() else []:
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            cached = self._summaries.get(p.stem)
            if not cached or cached[0] != mtime:
                run = self.load(p.stem)
                if not run:
                    continue
                cached = (mtime, self.summary(run))
                self._summaries[p.stem] = cached
            out.append(cached[1])
        return out

    # ── actions ──────────────────────────────────────────────────────────────
    def create(self, questions, mode: str, name: str = "", user: str | None = None) -> dict:
        if mode not in MODES:
            raise ValueError(f"Mode must be one of {', '.join(MODES)}.")
        items = []
        for q in questions if isinstance(questions, list) else []:
            if isinstance(q, str):
                q = {"question": q}
            if not isinstance(q, dict):
                continue
            text = str(q.get("question") or "").strip()[:MAX_CHARS]
            if not text:
                continue
            expected = str(q.get("expected") or "").strip()[:MAX_CHARS] or None
            items.append({"n": len(items) + 1, "question": text, "expected": expected,
                          "status": "pending", "answer": None, "sources": [], "retrieved": [],
                          "seconds": None, "error": None})
        if not items:
            raise ValueError("No questions given.")
        if len(items) > MAX_QUESTIONS:
            raise ValueError(f"{len(items)} questions — at most {MAX_QUESTIONS} in one run.")
        now = datetime.now()
        run = {"id": f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
               "name": str(name or "").strip()[:120] or f"Test {now:%d.%m.%Y %H:%M}",
               "mode": mode, "created_at": _now(), "started_by": user, "status": "queued",
               "started_at": None, "finished_at": None, "stop_reason": None, "items": items}
        self.save(run)
        self._enqueue(run["id"])
        return run

    def stop(self, rid: str) -> dict | None:
        """A queued run stops at once; a running one after its current question."""
        with self._lock:
            run = self.load(rid)
            if not run:
                return None
            if rid in self._queue:
                self._queue.remove(rid)
                run.update(status="stopped", stop_reason="stopped by an admin")
                self.save(run)
            elif rid == self._current:
                self._stop.add(rid)
            return run

    def resume(self, rid: str) -> dict:
        """Answer the questions not answered yet — the pending and the failed ones."""
        with self._lock:
            run = self.load(rid)
            if not run:
                raise LookupError("No such run.")
            if rid in self._queue or rid == self._current:
                raise ValueError("This run is already going.")
            if all(i.get("status") == "done" for i in run["items"]):
                raise ValueError("Every question of this run is answered.")
            run.update(status="queued", stop_reason=None, finished_at=None)
            self.save(run)
        self._enqueue(rid)
        return run

    def delete(self, rid: str) -> bool:
        with self._lock:
            if rid in self._queue or rid == self._current:
                raise ValueError("Stop the run before deleting it.")
            if not self.load(rid):
                return False
            self._path(rid).unlink(missing_ok=True)
            self._summaries.pop(rid, None)
            return True

    def is_active(self, rid: str) -> bool:
        with self._lock:
            return rid in self._queue or rid == self._current

    # ── the worker ───────────────────────────────────────────────────────────
    def _enqueue(self, rid: str) -> None:
        with self._lock:
            if rid not in self._queue and rid != self._current:
                self._queue.append(rid)
            self._stop.discard(rid)
            if not self._thread or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, name="batch-runs", daemon=True)
                self._thread.start()
        self._wake.set()

    def _worker(self) -> None:
        while True:
            with self._lock:
                rid = self._queue.pop(0) if self._queue else None
                self._current = rid
            if rid is None:
                self._wake.wait()
                self._wake.clear()
                continue
            try:
                self._run(rid)
            except Exception as e:          # a bug in one run must not end every later run
                print(f"[batch] run {rid} failed: {type(e).__name__}: {e}", flush=True)
                run = self.load(rid)
                if run:
                    run.update(status="stopped", stop_reason=f"internal error: {e}", finished_at=_now())
                    self.save(run)
            finally:
                with self._lock:
                    self._current = None
                    self._stop.discard(rid)

    def _run(self, rid: str) -> None:
        run = self.load(rid)
        if not run:
            return
        run.update(status="running", started_at=run.get("started_at") or _now(), stop_reason=None)
        self.save(run)
        print(f"[batch] run {rid} ({run['name']}): {len(run['items'])} questions, {run['mode']}", flush=True)
        failed_in_row = 0
        for it in run["items"]:
            if it.get("status") == "done":
                continue
            if rid in self._stop:
                run.update(status="stopped", stop_reason="stopped by an admin")
                break
            t0 = time.time()
            try:
                out = self.answer(it["question"], run["mode"]) or {}
                it.update(status="done", answer=out.get("answer") or "", sources=out.get("sources") or [],
                          retrieved=out.get("retrieved") or [], model=out.get("model"), error=None)
                failed_in_row = 0
            except Exception as e:
                it.update(status="error", answer=None, sources=[], retrieved=[],
                          error=(str(e) or type(e).__name__)[:500])
                failed_in_row += 1
            it["seconds"] = round(time.time() - t0, 1)
            self.save(run)
            if failed_in_row >= STOP_AFTER_ERRORS:
                run.update(status="stopped", stop_reason=f"{STOP_AFTER_ERRORS} questions in a row failed "
                                                         f"(is the bot running?) — last error: {it['error']}")
                break
        else:
            run["status"] = "done"
        run["finished_at"] = _now()
        self.save(run)
        print(f"[batch] run {rid} {run['status']}", flush=True)


# ── downloads ────────────────────────────────────────────────────────────────
def to_csv(run: dict) -> str:
    """UTF-8 with a BOM, so Excel shows Hindi. Two empty columns for the reviewer."""
    items = run.get("items") or []
    with_expected = any(i.get("expected") for i in items)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["No.", "Question"] + (["Expected answer"] if with_expected else []) +
               ["Bot answer", "Sources (cited)", "Documents retrieved", "Status", "Error", "Seconds",
                "Answer model", "Correct? (fill in)", "Notes (fill in)"])
    for i in items:
        sources = "\n".join(s.get("label") or "" for s in i.get("sources") or [])
        w.writerow([i.get("n"), i.get("question")] + ([i.get("expected") or ""] if with_expected else []) +
                   [i.get("answer") or "", sources, "\n".join(i.get("retrieved") or []),
                    i.get("status"), i.get("error") or "", i.get("seconds") if i.get("seconds") is not None else "",
                    i.get("model") or "", "", ""])
    return "﻿" + buf.getvalue()


def _filename(run: dict, ext: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", run.get("name") or "").strip("_")[:60]
    return f"batch-{run['id']}{'-' + stem if stem else ''}.{ext}"


# ── HTTP ─────────────────────────────────────────────────────────────────────
def blueprint(runs: Runs) -> Blueprint:
    """/api/batch/* — admin-only: the Node server lets only admins through
    (server.js), and Flask itself listens on loopback."""
    bp = Blueprint("batch_runs", __name__)

    def _missing():
        return jsonify({"success": False, "error": "No such run."}), 404

    @bp.route("/api/batch/runs", methods=["GET"])
    def batch_list():
        return jsonify({"success": True, "runs": runs.list(), "max_questions": MAX_QUESTIONS})

    @bp.route("/api/batch/runs", methods=["POST"])
    def batch_create():
        data = request.get_json(silent=True) or {}
        try:
            run = runs.create(data.get("questions"), str(data.get("mode") or "normal"),
                              str(data.get("name") or ""), request.headers.get("X-Gad-User") or None)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        return jsonify({"success": True, "run": runs.summary(run)}), 201

    @bp.route("/api/batch/runs/<rid>", methods=["GET"])
    def batch_get(rid):
        run = runs.load(rid)
        if not run:
            return _missing()
        return jsonify({"success": True, "run": run | runs.summary(run)})

    @bp.route("/api/batch/runs/<rid>/stop", methods=["POST"])
    def batch_stop(rid):
        run = runs.stop(rid)
        return jsonify({"success": True, "run": runs.summary(run)}) if run else _missing()

    @bp.route("/api/batch/runs/<rid>/resume", methods=["POST"])
    def batch_resume(rid):
        try:
            run = runs.resume(rid)
        except LookupError:
            return _missing()
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 409
        return jsonify({"success": True, "run": runs.summary(run)})

    @bp.route("/api/batch/runs/<rid>", methods=["DELETE"])
    def batch_delete(rid):
        try:
            ok = runs.delete(rid)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 409
        return jsonify({"success": True}) if ok else _missing()

    @bp.route("/api/batch/runs/<rid>/download", methods=["GET"])
    def batch_download(rid):
        run = runs.load(rid)
        if not run:
            return _missing()
        if request.args.get("format") == "json":
            body, mime, ext = json.dumps(run, ensure_ascii=False, indent=1), "application/json", "json"
        else:
            body, mime, ext = to_csv(run), "text/csv", "csv"
        return Response(body, mimetype=mime, headers={
            "Content-Disposition": f'attachment; filename="{_filename(run, ext)}"',
            "Cache-Control": "no-store"})

    return bp

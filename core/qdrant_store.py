"""The single owner of the vector store — the ONLY place a QdrantClient is built.

Why this module exists
──────────────────────
Embedded ("local") Qdrant is a single-process store. `QdrantLocal.__init__` takes an
`EXCLUSIVE | NON_BLOCKING` flock on `<store>/.lock`, so the *second* opener — another
thread, another process, a CLI run while the backend is up, or the tail of a restart
overlapping its replacement — dies with:

    Storage folder ... is already accessed by another instance of Qdrant client.

Before this module the project had seven independent `QdrantClient(path=...)` calls in
six files, across several processes, with nothing enforcing "one owner". Two openers
raced and lost, and the failure surfaced as a backend stuck on "Preparing…" or, worse,
as an empty retrieval that the answer path then papered over with the standing-facts
note. Retrying could not fix it, and made it worse: Qdrant loads the WHOLE collection
into RAM in `_load()` *before* it touches the lock (data/fg/db is 401MB / 26k points),
so every retry re-paid a full multi-second load and then failed instantly on a lock
that never blocks. Three retries meant three full loads and no progress.

The fix is structural, not another retry knob:

  * One client per (process, store). A process-wide, thread-safe singleton — so the
    Flask dev server's concurrent request threads and the background indexing threads
    all share one handle instead of racing to open one.
  * One process per store, decided BEFORE the expensive open. We take our own advisory
    lock (`<store>/.owner.lock`) first. Acquiring it is instant and costs no load, so a
    loser finds out in milliseconds instead of after a 401MB read, and a *waiter* waits
    on the lock itself — a real, cheap block, which Qdrant's non-blocking lock could
    never offer. That is what makes a restart race self-heal in well under a second.
  * A named holder. The winner records pid/command/mode in `<store>/.owner.json`, so a
    loser can say "held by pid 8123 (python app.py) since 14:02" instead of leaving an
    operator to guess. Stale entries need no reaping: the OS drops the flock when a
    process dies, so if the lock is free the old holder is gone by definition.
  * Failure stays loud. Callers get `StoreUnavailable` (with `.holder`), never a silent
    empty result that becomes a confident wrong answer.
  * Remote mode is first-class. `QDRANT_MODE=remote` skips the lock entirely — the
    server owns the store and concurrency stops being our problem — so escaping local
    mode's single-writer limit is a config flip, not a code change. See DEPLOY.md.

Usage
─────
    from qdrant_store import get_client, session, StoreUnavailable

    client = get_client()                     # long-lived services (Flask backend)

    with session(role="writer") as client:    # short-lived CLI tools
        ...                                   # lock released on exit
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock, Timeout
from qdrant_client import QdrantClient

# core/ is this file's directory, so pipeline_config is an ordinary sibling import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_config as pc  # noqa: E402

__all__ = [
    "StoreUnavailable",
    "StoreConfig",
    "config",
    "get_client",
    "session",
    "close_client",
    "close_all",
    "describe",
    "holder_of",
]

OWNER_LOCK_NAME = ".owner.lock"
OWNER_INFO_NAME = ".owner.json"

Role = Literal["reader", "writer"]


# ── Errors ─────────────────────────────────────────────────────────────────────


class StoreUnavailable(RuntimeError):
    """The vector store could not be opened.

    Raised instead of returning None or an empty result, because the two failures
    this replaces both used to masquerade as "no documents matched": a contended
    lock and an unreachable server. `holder` carries the owner record when the
    cause was contention, so callers and logs can name the process to stop.
    """

    def __init__(self, message: str, *, holder: dict[str, Any] | None = None):
        super().__init__(message)
        self.holder = holder


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoreConfig:
    """Resolved store settings. Env wins over the pipeline registry, as elsewhere."""

    mode: str
    path: Path
    collection: str
    host: str
    port: int
    api_key: str | None
    timeout: int
    lock_wait: float
    prefer_grpc: bool

    @property
    def is_remote(self) -> bool:
        return self.mode == "remote"

    @property
    def target(self) -> str:
        """Human-readable location, for logs and error messages."""
        return f"{self.host}:{self.port}" if self.is_remote else str(self.path)

    @property
    def key(self) -> str:
        """Identity of the store, so one process can hold several without collision."""
        return f"remote::{self.host}:{self.port}" if self.is_remote else f"local::{self.path}"


def config(**overrides: Any) -> StoreConfig:
    """Build a StoreConfig from env + pipeline registry, with explicit overrides.

    CHIPPY_QDRANT_LOCAL_PATH is honoured because the Node launcher sets it per
    pipeline (see selection/server/utils/pipelines.js) and it must keep winning.
    """
    mode = str(overrides.get("mode") or os.getenv("QDRANT_MODE", "local")).strip().lower()
    if mode not in ("local", "remote"):
        raise ValueError(f"QDRANT_MODE must be 'local' or 'remote', got {mode!r}")

    raw_path = overrides.get("path") or os.getenv("CHIPPY_QDRANT_LOCAL_PATH") or pc.paths.db
    collection = str(overrides.get("collection") or os.getenv("CHIPPY_QDRANT_COLLECTION") or pc.COLLECTION)
    if mode == "remote" and not collection.startswith(f"{pc.PIPELINE}__"):
        # One server holds every bot's collections. In local mode each bot had its
        # own directory, so they could all be named "db3" (pipelines.json); on a
        # shared server that name would be one collection for all three bots.
        collection = f"{pc.PIPELINE}__{collection}"
    return StoreConfig(
        mode=mode,
        path=Path(raw_path).expanduser().resolve(),
        collection=collection,
        host=str(overrides.get("host") or os.getenv("QDRANT_HOST", "localhost")),
        port=int(overrides.get("port") or os.getenv("QDRANT_PORT", "6333")),
        api_key=overrides.get("api_key") or os.getenv("QDRANT_API_KEY") or None,
        timeout=int(overrides.get("timeout") or os.getenv("QDRANT_TIMEOUT", "60")),
        # How long to block on the ownership lock. This is a cheap wait on a lock —
        # not the old load-then-fail retry loop — so a few seconds comfortably covers
        # the only benign case (a backend restart overlapping its predecessor) while
        # still failing fast enough that the UI never sits on "Preparing…".
        lock_wait=float(overrides.get("lock_wait") or os.getenv("QDRANT_LOCK_WAIT", "10")),
        prefer_grpc=_env_flag("QDRANT_PREFER_GRPC", default=False),
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Ownership ──────────────────────────────────────────────────────────────────


def _owner_record(cfg: StoreConfig, role: Role) -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "command": " ".join(sys.argv[:3]) or sys.executable,
        "pipeline": pc.PIPELINE,
        "collection": cfg.collection,
        "role": role,
        "mode": cfg.mode,
        "since": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def holder_of(path: Path | str) -> dict[str, Any] | None:
    """Who currently owns this local store, or None if nobody does.

    Reads the sidecar only when the lock is genuinely held, so a leftover
    `.owner.json` from a killed process is never reported as a live owner.
    """
    store = Path(path).expanduser().resolve()
    info_path = store / OWNER_INFO_NAME
    # A deliberately separate FileLock object, never the registered one: probing must
    # not reuse a lock this process already holds, or we would reentrantly acquire it
    # and report the store as free while we sit on it. Distinct objects mean distinct
    # descriptors, and both flock (POSIX) and msvcrt (Windows) conflict across
    # descriptors even within one process — so this answers correctly either way.
    probe = FileLock(str(store / OWNER_LOCK_NAME), thread_local=False)
    try:
        probe.acquire(timeout=0)
    except Timeout:
        try:
            return json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            # Held, but the sidecar is missing or corrupt — still better than "free".
            return {"pid": None, "command": "unknown", "since": "unknown"}
    except OSError:
        return None  # unreadable directory: report unknown rather than a false owner
    else:
        probe.release()
        return None


def _describe_holder(holder: dict[str, Any] | None) -> str:
    if not holder:
        return "another process"
    pid = holder.get("pid")
    command = holder.get("command") or "unknown"
    since = holder.get("since") or "unknown"
    return f"pid {pid} ({command}) since {since}"


def _external_holders(store: Path, budget: float | None = None) -> list[dict[str, Any]]:
    """Processes holding files inside the store WITHOUT going through this module.

    Ownership metadata only exists for openers that came through here. A store can
    still be held by something else — a backend started before this module existed,
    a stray script, an interactive session — and that was the hardest failure to
    diagnose, because Qdrant's own error names no one.

    Strictly time-boxed. `open_files()` is slow and permission-sensitive on Windows
    (a full sweep measured 181s on this machine), and a diagnostic that outlasts the
    failure it explains just becomes the next mystery hang. We spend a few seconds,
    return whatever we found, and let the message degrade to "could not identify it".
    psutil is optional; without it this simply returns nothing.
    """
    if budget is None:
        budget = float(os.getenv("QDRANT_HOLDER_SCAN_BUDGET", "5"))
    try:
        import psutil
    except Exception:
        return []

    found: list[dict[str, Any]] = []
    store_str = str(store).lower()
    me = os.getpid()
    deadline = time.monotonic() + budget
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > deadline:
                break
            if proc.info["pid"] == me:
                continue
            # Only Python processes can be holding an embedded Qdrant store, and
            # skipping everything else is what keeps the sweep inside its budget.
            name = (proc.info["name"] or "").lower()
            if "python" not in name and "uv" not in name:
                continue
            try:
                for handle in proc.open_files():
                    if handle.path.lower().startswith(store_str):
                        found.append({
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "command": " ".join((proc.info["cmdline"] or [])[:3]),
                        })
                        break
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
    except Exception:
        return found
    return found


# ── Handles ────────────────────────────────────────────────────────────────────


@dataclass
class _Handle:
    """One open store: the client plus the lock that entitles us to it."""

    cfg: StoreConfig
    client: QdrantClient
    lock: FileLock | None = None
    refs: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


# One FileLock object per path per process. Two FileLock instances over the same file
# in one process can both succeed (flock is per-descriptor), which would defeat the
# whole point — so intra-process exclusion rests on _STATE_LOCK plus this registry,
# and inter-process exclusion on the flock itself.
_LOCKS: dict[str, FileLock] = {}
_HANDLES: dict[str, _Handle] = {}
_STATE_LOCK = threading.RLock()


def _lock_for(cfg: StoreConfig) -> FileLock:
    lock = _LOCKS.get(cfg.key)
    if lock is None:
        cfg.path.mkdir(parents=True, exist_ok=True)
        # thread_local=False on purpose: ownership belongs to the PROCESS, not to the
        # thread that happened to open the store. With filelock's default the counter
        # and descriptor are thread-local, so a release from any other thread — the
        # main thread running atexit, say, after a request thread opened it — would
        # silently do nothing and leave the store locked for its whole lifetime.
        lock = FileLock(str(cfg.path / OWNER_LOCK_NAME), thread_local=False)
        _LOCKS[cfg.key] = lock
    return lock


def _acquire_ownership(cfg: StoreConfig, role: Role, wait: float) -> FileLock:
    """Claim the store, or explain who has it. Cheap: no collection load either way."""
    lock = _lock_for(cfg)
    try:
        lock.acquire(timeout=max(wait, 0))
    except Timeout:
        holder = holder_of(cfg.path)
        raise StoreUnavailable(
            f"Vector store {cfg.path} is in use by {_describe_holder(holder)}.\n"
            f"Embedded Qdrant allows one process at a time. Either stop that process, "
            f"or run a Qdrant server and set QDRANT_MODE=remote for concurrent access "
            f"(see DEPLOY.md).",
            holder=holder,
        ) from None

    try:
        (cfg.path / OWNER_INFO_NAME).write_text(
            json.dumps(_owner_record(cfg, role), indent=2), encoding="utf-8"
        )
    except OSError:
        pass  # advisory only — never fail an open because the sidecar is unwritable
    return lock


def _release_ownership(cfg: StoreConfig, lock: FileLock | None) -> None:
    if lock is None:
        return
    try:
        (cfg.path / OWNER_INFO_NAME).unlink(missing_ok=True)
    except OSError:
        pass
    if lock.is_locked:
        lock.release()


# ── Opening ────────────────────────────────────────────────────────────────────


def _qdrant_lock_is_free(store: Path) -> bool:
    """Is Qdrant's own `<store>/.lock` available right now?

    Uses portalocker with the exact flags QdrantLocal uses, so the answer means the
    same thing its constructor would conclude — just without loading the collection
    first. A missing lock file means nobody has ever opened this store. Any error
    reading it is reported as "free" so a diagnostic can never block a real open.
    """
    lock_path = store / ".lock"
    if not lock_path.exists():
        return True
    try:
        import portalocker
    except Exception:
        return True
    try:
        with open(lock_path, "r+") as handle:
            try:
                portalocker.lock(
                    handle,
                    portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING,
                )
            except portalocker.exceptions.LockException:
                return False
            portalocker.unlock(handle)
        return True
    except OSError:
        return True


def _external_lock_error(cfg: StoreConfig) -> StoreUnavailable:
    """Explain a holder that never came through this module, naming it when possible.

    This is the case Qdrant's own message leaves unexplained ("already accessed by
    another instance" — by whom?), and it was the hardest to diagnose in production.
    """
    outsiders = _external_holders(cfg.path)
    who = ", ".join(f"pid {h['pid']} ({h['command'] or h['name']})" for h in outsiders)
    return StoreUnavailable(
        f"Vector store {cfg.path} is locked by a process that did not go through "
        f"qdrant_store"
        + (f": {who}." if who else " (could not identify it).")
        + "\nThat is usually a backend still running from before this change, or a "
          "script opening the store directly. Stop it and retry, or run a Qdrant "
          "server and set QDRANT_MODE=remote for concurrent access (see DEPLOY.md).",
        holder={"external": outsiders} if outsiders else None,
    )


def _connect_remote(cfg: StoreConfig) -> QdrantClient:
    client = QdrantClient(
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
        timeout=cfg.timeout,
        prefer_grpc=cfg.prefer_grpc,
    )
    client.get_collections()  # fail here, at connect time, not mid-query
    return client


def _open(cfg: StoreConfig, role: Role, wait: float) -> _Handle:
    """Open the store for real. Callers must hold _STATE_LOCK."""
    if cfg.is_remote:
        print(f"[store] connecting to Qdrant server at {cfg.target}", flush=True)
        try:
            client = _connect_remote(cfg)
        except Exception as e:
            raise StoreUnavailable(
                f"Cannot reach the Qdrant server at {cfg.target}: {e}"
            ) from e
        print(f"[store] connected to {cfg.target}", flush=True)
        return _Handle(cfg=cfg, client=client, lock=None)

    lock = _acquire_ownership(cfg, role, wait)

    # Pre-flight Qdrant's OWN lock before paying for the open. QdrantLocal reads the
    # entire collection into RAM in _load() and only then takes a non-blocking flock,
    # so discovering contention inside the constructor costs a full load first —
    # measured at 204s against data/fg/db (26k points / 401MB), which is what made the
    # old three-attempt retry loop take ~10 minutes to report a one-line failure.
    # Probing the same lock file with the same semantics answers in microseconds.
    # (Protocol-aware openers are already excluded by the ownership lock above; this
    # covers everyone else — a backend from before this module, a stray script.)
    if not _qdrant_lock_is_free(cfg.path):
        _release_ownership(cfg, lock)
        raise _external_lock_error(cfg)

    t0 = time.time()
    print(f"[store] opening embedded Qdrant at {cfg.target} ({role})", flush=True)
    try:
        client = QdrantClient(path=str(cfg.path))
        client.get_collections()
    except Exception as e:
        # Never leave the store marked as ours when we failed to open it — that would
        # lock everyone else out of a store nobody is using.
        _release_ownership(cfg, lock)
        if "already accessed by another instance" in str(e):
            # Lost the narrow race between the pre-flight probe and this open.
            raise _external_lock_error(cfg) from e
        raise StoreUnavailable(f"Cannot open the vector store at {cfg.path}: {e}") from e

    print(f"[store] ready in {time.time() - t0:.1f}s", flush=True)
    return _Handle(cfg=cfg, client=client, lock=lock)


def ensure_indexes(client: QdrantClient, collection: str) -> None:
    """Payload indexes for the fields the bots filter on (idempotent). On a Qdrant
    server a filter on an unindexed field scans every point: measured at 85,000
    chunks, an answer's neighbour lookup (source + chunk) took 5.8 s without these.
    Embedded (local) Qdrant ignores payload indexes, so this is harmless there."""
    from qdrant_client import models as m
    wanted = [("source", m.PayloadSchemaType.KEYWORD), ("chunk", m.PayloadSchemaType.KEYWORD),
              ("doc_id", m.PayloadSchemaType.KEYWORD), ("chunk_type", m.PayloadSchemaType.KEYWORD),
              ("text", m.TextIndexParams(type="text", tokenizer=m.TokenizerType.WORD,
                                         min_token_len=2, lowercase=True))]
    try:
        have = set((client.get_collection(collection).payload_schema or {}).keys())
    except Exception:          # noqa: BLE001 — no collection yet / local mode: try anyway
        have = set()
    for field, schema in wanted:
        if field in have:
            continue
        try:
            client.create_payload_index(collection, field, field_schema=schema)
        except Exception as e:  # noqa: BLE001 — a missing index costs speed, not answers
            print(f"[store] payload index on '{field}' not created: {e}", flush=True)


def get_client(*, role: Role = "reader", wait: float | None = None, **overrides: Any) -> QdrantClient:
    """The process's client for this store, opened on first use.

    Idempotent and thread-safe: concurrent callers that all see "no client yet" are
    serialised here and exactly one of them opens the store. This is the invariant
    long-lived services rely on — the Flask backend serves requests on multiple
    threads and indexes documents on background threads, all sharing this handle.

    Raises StoreUnavailable if the store is owned elsewhere or unreachable.
    """
    cfg = config(**overrides)
    with _STATE_LOCK:
        handle = _HANDLES.get(cfg.key)
        if handle is not None:
            return handle.client
        handle = _open(cfg, role, cfg.lock_wait if wait is None else wait)
        _HANDLES[cfg.key] = handle
        return handle.client


class session:
    """Scoped access for short-lived tools, releasing the store on exit.

    A CLI that holds the store for its whole run blocks the backend from restarting,
    so tools should scope their access. Nests safely: an inner session over a store
    the process already owns reuses the open client and does not close it on exit.
    """

    def __init__(self, *, role: Role = "reader", wait: float | None = None, **overrides: Any):
        self._cfg = config(**overrides)
        self._role = role
        self._wait = self._cfg.lock_wait if wait is None else wait
        self._owns = False

    def __enter__(self) -> QdrantClient:
        with _STATE_LOCK:
            handle = _HANDLES.get(self._cfg.key)
            if handle is None:
                handle = _open(self._cfg, self._role, self._wait)
                _HANDLES[self._cfg.key] = handle
                self._owns = True
            handle.refs += 1
            return handle.client

    def __exit__(self, *exc: Any) -> Literal[False]:
        with _STATE_LOCK:
            handle = _HANDLES.get(self._cfg.key)
            if handle is not None:
                handle.refs -= 1
                if self._owns and handle.refs <= 0:
                    _close(self._cfg.key)
        return False


# ── Closing ────────────────────────────────────────────────────────────────────


def _close(key: str) -> None:
    """Drop one handle. Callers must hold _STATE_LOCK."""
    handle = _HANDLES.pop(key, None)
    if handle is None:
        return
    try:
        handle.client.close()
    except Exception:
        pass  # a half-closed client must not stop us from freeing the lock
    _release_ownership(handle.cfg, handle.lock)


def close_client(**overrides: Any) -> None:
    """Close this store and hand ownership back, so another process can take it."""
    cfg = config(**overrides)
    with _STATE_LOCK:
        _close(cfg.key)


def close_all() -> None:
    with _STATE_LOCK:
        for key in list(_HANDLES):
            _close(key)


# Registered once, at import. Releases the flock and clears `.owner.json` on a clean
# exit; on a hard kill the OS releases the flock anyway and `holder_of` ignores the
# orphaned sidecar, so a crash cannot wedge the store either way.
atexit.register(close_all)


# ── Diagnostics ────────────────────────────────────────────────────────────────


def describe(**overrides: Any) -> dict[str, Any]:
    """Everything an operator needs to explain a store problem, without opening it.

    Deliberately non-invasive: it reports whether THIS process holds the store and
    who else might, so a health endpoint can surface "held by pid N" rather than the
    old symptom of an endless "Preparing…".
    """
    cfg = config(**overrides)
    with _STATE_LOCK:
        handle = _HANDLES.get(cfg.key)
        open_here = handle is not None

    info: dict[str, Any] = {
        "mode": cfg.mode,
        "target": cfg.target,
        "collection": cfg.collection,
        "open_in_this_process": open_here,
        "pid": os.getpid(),
        "holder": None,
        "collection_exists": None,
        "points": None,
        "error": None,
    }
    if not cfg.is_remote and not open_here:
        info["holder"] = holder_of(cfg.path)

    if open_here and handle is not None:
        try:
            exists = handle.client.collection_exists(cfg.collection)
            info["collection_exists"] = exists
            if exists:
                info["points"] = handle.client.count(cfg.collection).count
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
    return info


def doctor(pipelines: list[str] | None = None) -> int:
    """Report every pipeline's store and who owns it. Returns a shell exit code.

    Answers "why won't it start / why is a query failing" without opening anything,
    so running the doctor can never itself take the lock it is reporting on.
    """
    registry = json.loads((pc.REPO_ROOT / "pipelines.json").read_text(encoding="utf-8"))
    names = pipelines or list(registry)
    busy = 0

    print(f"QDRANT_MODE={os.getenv('QDRANT_MODE', 'local')}   "
          f"RAG_DATA_ROOT={os.getenv('RAG_DATA_ROOT', '<repo>/data')}")
    for name in names:
        if name not in registry:
            print(f"\n{name}: unknown pipeline (known: {', '.join(registry)})")
            busy += 1
            continue
        store = pc.DATA_BASE / registry[name]["dataRoot"] / "db"
        cfg = config(path=store, collection=registry[name]["collection"])
        print(f"\n{name}  ({registry[name].get('shortLabel', name)})")
        print(f"  mode       : {cfg.mode}")
        print(f"  store      : {cfg.target}")
        print(f"  collection : {cfg.collection}")
        if cfg.is_remote:
            print("  owner      : n/a (the server owns the store; concurrent access is fine)")
            continue
        if not store.exists():
            print("  owner      : store not built yet (run scripts/ingest_corpus.py)")
            continue

        owner = holder_of(store)
        if owner:
            print(f"  owner      : {_describe_holder(owner)}  [via qdrant_store]")
            busy += 1
        elif not _qdrant_lock_is_free(store):
            outsiders = _external_holders(store)
            who = ", ".join(f"pid {h['pid']} ({h['command'] or h['name']})" for h in outsiders)
            print(f"  owner      : {who or 'unidentified process'}  "
                  f"[holds Qdrant's own lock, outside qdrant_store]")
            busy += 1
        else:
            print("  owner      : free")
    return 1 if busy else 0


if __name__ == "__main__":  # `python core/qdrant_store.py [pipeline ...]` — store doctor
    sys.exit(doctor(sys.argv[1:] or None))

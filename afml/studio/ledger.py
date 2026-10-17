"""
Research ledger: one SQLite database that answers "how many things did I try to
get this number?"

Tables
------
families      one research question each, the unit over which N is counted
runs          every launched run: family, stage, config hash, code version, status
jobs          the queue the worker consumes (one row per run)
trials        every scored attempt; a grid or Optuna study adds one row per candidate
fold_scores   per-trial, per-fold scores, the matrix effective_trial_count consumes
data_access   symbol, date range and purpose of every bar load
artifacts     path, kind and content hash of every file a run writes
research_end  per-symbol hold-out boundary (see vault.py)
preregistrations, holdout_access   hold-out bookkeeping (see vault.py)
workers       worker heartbeats, so the UI can tell whether a worker is alive
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from . import paths
from .stats import effective_trial_count, expected_max_sharpe, noise_floor

SCHEMA = """
CREATE TABLE IF NOT EXISTS families (
    family_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    contaminated INTEGER NOT NULL DEFAULT 0,
    contaminated_at TEXT,
    contaminated_reason TEXT,
    frozen_config_hash TEXT,
    closed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL REFERENCES families(family_id),
    stage TEXT NOT NULL,
    purpose TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_version TEXT NOT NULL,
    identity TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    parent_runs TEXT NOT NULL DEFAULT '[]',
    run_dir TEXT NOT NULL,
    error TEXT,
    summary_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS runs_family ON runs(family_id);
CREATE INDEX IF NOT EXISTS runs_identity ON runs(identity);
CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    status TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    worker_pid INTEGER,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    family_id TEXT NOT NULL REFERENCES families(family_id),
    kind TEXT NOT NULL,
    number INTEGER NOT NULL DEFAULT 0,
    metric TEXT,
    score REAL,
    sharpe REAL,
    params_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trials_family ON trials(family_id);
CREATE TABLE IF NOT EXISTS fold_scores (
    trial_id INTEGER NOT NULL REFERENCES trials(trial_id),
    fold INTEGER NOT NULL,
    score REAL,
    PRIMARY KEY (trial_id, fold)
);
CREATE TABLE IF NOT EXISTS data_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    family_id TEXT,
    symbol TEXT NOT NULL,
    start TEXT,
    end TEXT,
    purpose TEXT NOT NULL,
    n_rows INTEGER,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    PRIMARY KEY (run_id, path)
);
CREATE TABLE IF NOT EXISTS research_end (
    symbol TEXT PRIMARY KEY,
    research_end TEXT NOT NULL,
    set_at TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS preregistrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id TEXT NOT NULL REFERENCES families(family_id),
    config_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS workers (
    pid INTEGER PRIMARY KEY,
    host TEXT,
    started_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    status TEXT NOT NULL
);
"""

ACTIVE_STATUSES = ("queued", "running")
FINAL_STATUSES = ("done", "failed", "cancelled")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else paths.ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: Iterable = ()) -> List[Dict[str, Any]]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ families

    def upsert_family(self, family_id: str, name: str, key: Dict[str, Any]) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT OR IGNORE INTO families(family_id, name, key_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (family_id, name, json.dumps(key, sort_keys=True), utcnow()),
            )

    def get_family(self, family_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM families WHERE family_id = ?", (family_id,))

    def list_families(self) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM families ORDER BY created_at DESC")

    def mark_contaminated(self, family_id: str, reason: str) -> None:
        """Permanent: there is no call that clears the flag."""
        with self.transaction() as con:
            con.execute(
                "UPDATE families SET contaminated = 1, contaminated_at = COALESCE(contaminated_at, ?), "
                "contaminated_reason = COALESCE(contaminated_reason, ?) WHERE family_id = ?",
                (utcnow(), reason, family_id),
            )

    # ------------------------------------------------------------------ runs and jobs

    def create_run(self, *, run_id: str, family_id: str, stage: str, purpose: str,
                   config_hash: str, code_version: str, identity: str, run_dir: str,
                   parent_runs: Optional[List[str]] = None, notes: str = "",
                   enqueue: bool = True) -> None:
        now = utcnow()
        with self.transaction() as con:
            con.execute(
                "INSERT INTO runs(run_id, family_id, stage, purpose, config_hash, code_version, "
                "identity, status, created_at, parent_runs, run_dir, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                (run_id, family_id, stage, purpose, config_hash, code_version, identity, now,
                 json.dumps(parent_runs or []), run_dir, notes),
            )
            if enqueue:
                con.execute(
                    "INSERT INTO jobs(run_id, status, submitted_at) VALUES (?, 'queued', ?)",
                    (run_id, now),
                )

    def update_run(self, run_id: str, **fields: Any) -> None:
        if "summary" in fields:
            fields["summary_json"] = json.dumps(fields.pop("summary"), default=str)
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction() as con:
            con.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.one(
            "SELECT r.*, j.pid, j.cancel_requested, j.exit_code, f.name AS family_name, "
            "f.contaminated FROM runs r LEFT JOIN jobs j USING(run_id) "
            "LEFT JOIN families f USING(family_id) WHERE r.run_id = ?",
            (run_id,),
        )
        return _decode_run(row) if row else None

    def list_runs(self, *, family_id: Optional[str] = None, stage: Optional[str] = None,
                  statuses: Optional[Iterable[str]] = None, limit: int = 200) -> List[Dict[str, Any]]:
        where, params = [], []
        if family_id:
            where.append("r.family_id = ?")
            params.append(family_id)
        if stage:
            where.append("r.stage = ?")
            params.append(stage)
        if statuses:
            statuses = list(statuses)
            where.append(f"r.status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        sql = (
            "SELECT r.*, j.pid, j.cancel_requested, f.name AS family_name, f.contaminated "
            "FROM runs r LEFT JOIN jobs j USING(run_id) LEFT JOIN families f USING(family_id)"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT ?"
        params.append(limit)
        return [_decode_run(r) for r in self.query(sql, params)]

    def set_status(self, run_id: str, status: str, *, error: Optional[str] = None,
                   exit_code: Optional[int] = None) -> None:
        now = utcnow()
        with self.transaction() as con:
            started = now if status == "running" else None
            finished = now if status in FINAL_STATUSES else None
            con.execute(
                "UPDATE runs SET status = ?, error = COALESCE(?, error), "
                "started_at = COALESCE(started_at, ?), finished_at = COALESCE(?, finished_at) "
                "WHERE run_id = ?",
                (status, error, started, finished, run_id),
            )
            con.execute(
                "UPDATE jobs SET status = ?, exit_code = COALESCE(?, exit_code), "
                "started_at = COALESCE(started_at, ?), finished_at = COALESCE(?, finished_at) "
                "WHERE run_id = ?",
                (status, exit_code, started, finished, run_id),
            )

    def claim_next_job(self, worker_pid: int) -> Optional[str]:
        """Atomically move the oldest queued job to running and return its run ID."""
        with self.transaction() as con:
            row = con.execute(
                "SELECT run_id FROM jobs WHERE status = 'queued' AND cancel_requested = 0 "
                "ORDER BY submitted_at, rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utcnow()
            con.execute(
                "UPDATE jobs SET status = 'running', worker_pid = ?, started_at = ? WHERE run_id = ?",
                (worker_pid, now, row["run_id"]),
            )
            con.execute(
                "UPDATE runs SET status = 'running', started_at = ? WHERE run_id = ?",
                (now, row["run_id"]),
            )
            return row["run_id"]

    def set_job_pid(self, run_id: str, pid: Optional[int]) -> None:
        with self.transaction() as con:
            con.execute("UPDATE jobs SET pid = ? WHERE run_id = ?", (pid, run_id))

    def request_cancel(self, run_id: str) -> None:
        with self.transaction() as con:
            con.execute("UPDATE jobs SET cancel_requested = 1 WHERE run_id = ?", (run_id,))

    def get_job(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM jobs WHERE run_id = ?", (run_id,))

    def find_identity(self, identity: str) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT run_id, status, created_at FROM runs WHERE identity = ? ORDER BY created_at",
            (identity,),
        )

    # ------------------------------------------------------------------ trials

    def add_trial(self, *, run_id: str, family_id: str, kind: str, number: int = 0,
                  metric: Optional[str] = None, score: Optional[float] = None,
                  sharpe: Optional[float] = None, params: Optional[Dict] = None,
                  fold_scores: Optional[Iterable[float]] = None) -> int:
        with self.transaction() as con:
            cur = con.execute(
                "INSERT INTO trials(run_id, family_id, kind, number, metric, score, sharpe, "
                "params_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, family_id, kind, number, metric, _f(score), _f(sharpe),
                 json.dumps(params, default=str) if params is not None else None, utcnow()),
            )
            trial_id = cur.lastrowid
            if fold_scores is not None:
                con.executemany(
                    "INSERT INTO fold_scores(trial_id, fold, score) VALUES (?, ?, ?)",
                    [(trial_id, k, _f(s)) for k, s in enumerate(fold_scores)],
                )
            return trial_id

    def trials(self, family_id: str) -> List[Dict[str, Any]]:
        return self.query(
            "SELECT * FROM trials WHERE family_id = ? ORDER BY trial_id", (family_id,)
        )

    def score_matrix(self, family_id: str, metric: Optional[str] = None) -> np.ndarray:
        """Trials x folds matrix over trials with the family's most common fold count."""
        sql = ("SELECT t.trial_id, fs.fold, fs.score FROM trials t JOIN fold_scores fs "
               "USING(trial_id) WHERE t.family_id = ?")
        params: List[Any] = [family_id]
        if metric is not None:
            sql += " AND t.metric = ?"
            params.append(metric)
        rows = self.query(sql + " ORDER BY t.trial_id, fs.fold", params)
        by_trial: Dict[int, List[float]] = {}
        for r in rows:
            by_trial.setdefault(r["trial_id"], []).append(
                np.nan if r["score"] is None else r["score"]
            )
        if not by_trial:
            return np.empty((0, 0))
        lengths = [len(v) for v in by_trial.values()]
        modal = max(set(lengths), key=lengths.count)
        return np.array([v for v in by_trial.values() if len(v) == modal], dtype=float)

    def family_stats(self, family_id: str, metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Everything the UI shows beside a score: raw N, eigenvalue and Kish effective
        counts, the rank-limited flag, the best score with its noise floor, and the
        null threshold (expected best score under no skill at the effective N).

        ``n_raw`` counts every trial in the family whatever its metric. Scores are only
        comparable within one metric, so the score-based numbers use ``metric`` (default:
        the metric with the most scored trials).
        """
        trials = self.trials(family_id)
        all_scored = [t for t in trials if t["score"] is not None]
        metrics = sorted({t["metric"] for t in all_scored})
        if metric is None and all_scored:
            counts = {m: sum(t["metric"] == m for t in all_scored) for m in metrics}
            metric = max(metrics, key=lambda m: counts[m])
        scored = [t for t in all_scored if t["metric"] == metric]
        out: Dict[str, Any] = {
            "n_raw": len(trials),
            "n_scored": len(all_scored),
            "metric": metric,
            "metrics": metrics,
            "n_metric": len(scored),
            "n_eigen": None,
            "n_kish": None,
            "n_obs": None,
            "rank_limited": False,
            "best_score": None,
            "best_metric": None,
            "best_run": None,
            "best_noise_floor": None,
            "score_variance": None,
            "null_threshold": None,
        }
        mat = self.score_matrix(family_id, metric)
        if mat.size:
            tc = effective_trial_count(mat)
            out.update(n_eigen=tc.n_eigen, n_kish=tc.n_kish, n_obs=tc.n_obs,
                       rank_limited=tc.rank_limited)
        if scored:
            best = max(scored, key=lambda t: t["score"])
            out.update(best_score=best["score"], best_metric=best["metric"],
                       best_run=best["run_id"])
            folds = self.query("SELECT score FROM fold_scores WHERE trial_id = ?",
                               (best["trial_id"],))
            out["best_noise_floor"] = noise_floor([f["score"] for f in folds if f["score"] is not None])
            scores = np.array([t["score"] for t in scored], dtype=float)
            if len(scores) > 1:
                var = float(scores.var(ddof=1))
                n_eff = out["n_eigen"] or len(scores)
                out["score_variance"] = var
                out["null_threshold"] = expected_max_sharpe(n_eff, var, float(scores.mean()))
        return out

    # ------------------------------------------------------------------ data access, artifacts

    def add_data_access(self, *, symbol: str, start: Any, end: Any, purpose: str,
                        run_id: Optional[str] = None, family_id: Optional[str] = None,
                        n_rows: Optional[int] = None) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT INTO data_access(run_id, family_id, symbol, start, end, purpose, n_rows, at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, family_id, symbol, str(start), str(end), purpose, n_rows, utcnow()),
            )

    def add_artifact(self, run_id: str, path: str, kind: str, sha256: str, size: int) -> None:
        with self.transaction() as con:
            con.execute(
                "INSERT OR REPLACE INTO artifacts(run_id, path, kind, sha256, bytes) VALUES (?, ?, ?, ?, ?)",
                (run_id, path, kind, sha256, size),
            )

    def artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        return self.query("SELECT * FROM artifacts WHERE run_id = ? ORDER BY path", (run_id,))

    # ------------------------------------------------------------------ workers

    def heartbeat(self, pid: int, host: str, status: str = "idle") -> None:
        now = utcnow()
        with self.transaction() as con:
            con.execute(
                "INSERT INTO workers(pid, host, started_at, heartbeat_at, status) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(pid) DO UPDATE SET heartbeat_at = excluded.heartbeat_at, status = excluded.status",
                (pid, host, now, now, status),
            )

    def live_workers(self, max_age_s: float = 15.0) -> List[Dict[str, Any]]:
        rows = self.query("SELECT * FROM workers WHERE status != 'stopped'")
        now = time.time()
        live = []
        for r in rows:
            age = now - datetime.fromisoformat(r["heartbeat_at"]).timestamp()
            if age <= max_age_s:
                r["age_s"] = age
                live.append(r)
        return live


def _f(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _decode_run(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["parent_runs"] = json.loads(row.get("parent_runs") or "[]")
    row["summary"] = json.loads(row["summary_json"]) if row.get("summary_json") else {}
    return row


_default: Optional[Ledger] = None


def get_ledger() -> Ledger:
    """The ledger under the current studio home (re-resolved if the home changes)."""
    global _default
    path = paths.ledger_path()
    if _default is None or _default.path != path:
        _default = Ledger(path)
    return _default

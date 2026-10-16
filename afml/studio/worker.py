"""
The studio worker: ``python -m afml.studio.worker``.

Consumes the ledger's jobs table and launches one subprocess per job
(``python -m afml.studio.runner <run_id>``). Concurrency defaults to one heavy job,
because ``n_jobs=-1`` inside the pipelines already saturates the cores; later jobs
queue. The worker never imports pipeline code itself.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from afml.logging_config import configure_logging

from . import paths
from .jobs import is_alive, kill_tree
from .ledger import Ledger, get_ledger


def _spawn(run_id: str, run_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    root = str(paths.repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The studio home is an absolute path the child must share
    env["AFML_STUDIO_HOME"] = str(paths.studio_home())
    # afml.logging_config writes a file log on import; keep it out of the checkout
    env.setdefault("AFML_LOG_DIR", str(paths.studio_home() / "logs"))
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    out = open(run_dir / "process.log", "ab")
    return subprocess.Popen(
        [sys.executable, "-m", "afml.studio.runner", run_id],
        cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
    )


def spawn_detached(concurrency: int = 1) -> int:
    """Start a worker that outlives the caller (e.g. the Streamlit process). Returns its PID."""
    env = os.environ.copy()
    root = str(paths.repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["AFML_STUDIO_HOME"] = str(paths.studio_home())
    env.setdefault("AFML_LOG_DIR", str(paths.studio_home() / "logs"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    flags = 0
    if os.name == "nt":
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)
    log = open(paths.studio_home() / "worker.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "afml.studio.worker", "--concurrency", str(concurrency)],
        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )
    return proc.pid


def recover_orphans(ledger: Ledger) -> int:
    """Mark runs left 'running' by a dead process as failed."""
    n = 0
    for run in ledger.list_runs(statuses=["running"], limit=1000):
        if not is_alive(run.get("pid")):
            ledger.set_status(run["run_id"], "failed", error="process died (worker restarted)")
            n += 1
    return n


def run_worker(concurrency: int = 1, poll_s: float = 1.0, once: bool = False,
               ledger: Optional[Ledger] = None) -> None:
    ledger = ledger or get_ledger()
    pid, host = os.getpid(), socket.gethostname()
    running: Dict[str, subprocess.Popen] = {}
    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)

    n = recover_orphans(ledger)
    if n:
        logger.warning("marked {} orphaned run(s) failed", n)
    logger.info("worker {} on {} (concurrency {}) watching {}", pid, host, concurrency, ledger.path)

    try:
        while not stopping:
            ledger.heartbeat(pid, host, "busy" if running else "idle")

            # Reap finished processes and honour cancel requests
            for run_id, proc in list(running.items()):
                job = ledger.get_job(run_id) or {}
                if job.get("cancel_requested") and proc.poll() is None:
                    kill_tree(proc.pid)
                rc = proc.poll()
                if rc is None:
                    continue
                del running[run_id]
                run = ledger.get_run(run_id)
                if job.get("cancel_requested"):
                    ledger.set_status(run_id, "cancelled", error="cancelled by user", exit_code=rc)
                elif run and run["status"] == "running":
                    ledger.set_status(run_id, "failed", exit_code=rc,
                                      error=f"runner exited with code {rc}; see process.log")
                logger.info("run {} finished: {}", run_id, (ledger.get_run(run_id) or {}).get("status"))

            # Fill free slots
            while len(running) < concurrency and not stopping:
                run_id = ledger.claim_next_job(pid)
                if run_id is None:
                    break
                run = ledger.get_run(run_id)
                proc = _spawn(run_id, Path(run["run_dir"]))
                ledger.set_job_pid(run_id, proc.pid)
                running[run_id] = proc
                logger.info("run {} started as pid {}", run_id, proc.pid)

            if once and not running:
                queued = ledger.one("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued' "
                                    "AND cancel_requested = 0")["n"]
                if not queued:
                    break
            time.sleep(poll_s)
    finally:
        for run_id, proc in running.items():
            if proc.poll() is None:
                kill_tree(proc.pid)
                ledger.set_status(run_id, "failed", error="worker stopped while running")
        ledger.heartbeat(pid, host, "stopped")
        logger.info("worker {} stopped", pid)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m afml.studio.worker")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="exit when the queue is empty")
    args = ap.parse_args(argv)
    configure_logging()
    run_worker(args.concurrency, args.poll, args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
``python -m afml.studio <command>``

  run <config.json>              submit and execute now (logged as a new trial)
  submit <config.json>           queue for the worker
  worker [--once]                start the worker
  ls [--family F] [--stage S]    list runs
  show <run_id>                  run details, progress and summary
  log <run_id>                   tail a run's log
  cancel <run_id>                cancel a queued or running job
  resume <run_id>                resubmit as a child run
  families                       family scoreboard
  set-research-end SYMBOL DATE   set a symbol's hold-out boundary (once)
  research-ends                  list hold-out boundaries
  preregister FAMILY CONFIG_HASH pre-register a frozen config for hold-out
  example [--stage S]            print an example config
  app [--address A] [--port P]   launch the Streamlit app
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

from . import jobs, paths, vault
from .ledger import get_ledger


def example_config(stage: str = "importance", symbol: str = "SYNTH") -> dict:
    cfg = {
        "stage": stage,
        "purpose": "research",
        "data": {
            "symbol": symbol,
            "source": "synthetic",
            "source_params": {"seed": 7, "freq": "15min", "signal_strength": 0.05},
            "bar_type": "time",
            "bar_size": "M15",
            "price": "mid_price",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        },
        "labels": {
            "strategy": {"name": "bollinger", "params": {"window": 20, "num_std": 1.5}},
            "target": {"name": "daily_vol", "params": {"lookback": 50}},
            "profit_target": 1.0,
            "stop_loss": 1.0,
            "max_holding_period": {"hours": 12},
        },
        "features": {
            "steps": [
                {"name": "log_returns", "params": {"lags": [1, 4, 16]}},
                {"name": "volatility", "params": {"window": 20}},
                {"name": "frac_diff_close", "params": {"d": 0.4}},
            ]
        },
        "importance": {
            "methods": ["mdi", "mda"],
            "n_splits": 4,
            "estimator": {"name": "random_forest", "params": {"n_estimators": 50, "n_jobs": 1}},
        },
        "model": {
            "estimator": {"name": "random_forest", "params": {"n_estimators": 50, "n_jobs": 1}},
            "param_grid": {"max_depth": [3, 6]},
            "cv_splits": 4,
        },
    }
    from .schemas import STAGE_SECTIONS

    keep = set(STAGE_SECTIONS[stage]) | {"stage", "purpose"}
    return {k: v for k, v in cfg.items() if k in keep}


def _print_table(rows: List[dict], cols: List[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "") if r.get(c) is not None else "").ljust(widths[c]) for c in cols))


def _fmt(x, nd=4):
    return None if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else x)


def cmd_run(args) -> int:
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = jobs.submit(spec, enqueue=True)
    print(f"run {run_id} submitted; executing now")
    ledger = get_ledger()
    # Claim it ourselves so a running worker does not pick it up too
    with ledger.transaction() as con:
        con.execute("UPDATE jobs SET status = 'running', worker_pid = -1 WHERE run_id = ?", (run_id,))
    from .worker import _spawn

    run = ledger.get_run(run_id)
    proc = _spawn(run_id, Path(run["run_dir"]))
    ledger.set_job_pid(run_id, proc.pid)
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        jobs.cancel(run_id)
        print("cancelled")
        return 130
    run = ledger.get_run(run_id)
    if run["status"] == "running":
        ledger.set_status(run_id, "failed", exit_code=rc, error=f"runner exited with code {rc}")
        run = ledger.get_run(run_id)
    print(f"status: {run['status']}  run folder: {run['run_dir']}")
    if run["error"]:
        print(f"error: {run['error']}")
    return 0 if run["status"] == "done" else 1


def cmd_submit(args) -> int:
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(jobs.submit(spec))
    return 0


def cmd_worker(args) -> int:
    from afml.logging_config import configure_logging

    from .worker import run_worker

    configure_logging()
    run_worker(args.concurrency, args.poll, args.once)
    return 0


def cmd_ls(args) -> int:
    rows = get_ledger().list_runs(family_id=args.family, stage=args.stage, limit=args.limit)
    _print_table(rows, ["run_id", "stage", "purpose", "status", "family_name", "created_at"])
    return 0


def cmd_show(args) -> int:
    st = jobs.status(args.run_id)
    st.pop("summary_json", None)
    print(json.dumps(st, indent=2, default=str))
    return 0


def cmd_log(args) -> int:
    run = get_ledger().get_run(args.run_id)
    for rec in jobs.read_log(run["run_dir"], tail=args.tail):
        print(f"{rec['time'][:19]} {rec['level']:<8} [{rec['stage']}] {rec['message']}")
    return 0


def cmd_cancel(args) -> int:
    print(jobs.cancel(args.run_id))
    return 0


def cmd_resume(args) -> int:
    print(jobs.resume(args.run_id))
    return 0


def cmd_families(args) -> int:
    ledger = get_ledger()
    rows = []
    for f in ledger.list_families():
        s = ledger.family_stats(f["family_id"])
        rows.append({
            "family_id": f["family_id"], "name": f["name"], "N": s["n_raw"],
            "N_eigen": _fmt(s["n_eigen"], 1), "N_kish": _fmt(s["n_kish"], 1),
            "best": _fmt(s["best_score"]), "null": _fmt(s["null_threshold"]),
            "flags": ("CONTAMINATED " if f["contaminated"] else "") + ("rank-limited" if s["rank_limited"] else ""),
        })
    _print_table(rows, ["family_id", "name", "N", "N_eigen", "N_kish", "best", "null", "flags"])
    return 0


def cmd_set_research_end(args) -> int:
    d = vault.set_research_end(args.symbol, args.date, note=args.note or "")
    print(f"{args.symbol}: research_end = {d}")
    return 0


def cmd_research_ends(args) -> int:
    rows = [{"symbol": s, "research_end": d} for s, d in vault.all_research_ends().items()]
    _print_table(rows, ["symbol", "research_end"])
    return 0


def cmd_preregister(args) -> int:
    vault.preregister(args.family_id, args.config_hash)
    print("pre-registered")
    return 0


def cmd_example(args) -> int:
    print(json.dumps(example_config(args.stage, args.symbol), indent=2))
    return 0


def cmd_app(args) -> int:
    home = paths.repo_root() / "app" / "Home.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(home),
           "--server.address", args.address, "--server.port", str(args.port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false",
           "--theme.base", "light"]
    env = os.environ.copy()
    env.setdefault("AFML_LOG_DIR", str(paths.studio_home() / "logs"))
    return subprocess.call(cmd, cwd=str(paths.repo_root()), env=env)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m afml.studio", description="AFML Studio core")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run"); p.add_argument("config"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit"); p.add_argument("config"); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("worker")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--poll", type=float, default=1.0)
    p.add_argument("--once", action="store_true")
    p.set_defaults(fn=cmd_worker)
    p = sub.add_parser("ls")
    p.add_argument("--family"); p.add_argument("--stage"); p.add_argument("--limit", type=int, default=50)
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("show"); p.add_argument("run_id"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("log"); p.add_argument("run_id"); p.add_argument("--tail", type=int, default=50)
    p.set_defaults(fn=cmd_log)
    p = sub.add_parser("cancel"); p.add_argument("run_id"); p.set_defaults(fn=cmd_cancel)
    p = sub.add_parser("resume"); p.add_argument("run_id"); p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("families"); p.set_defaults(fn=cmd_families)
    p = sub.add_parser("set-research-end")
    p.add_argument("symbol"); p.add_argument("date"); p.add_argument("--note")
    p.set_defaults(fn=cmd_set_research_end)
    p = sub.add_parser("research-ends"); p.set_defaults(fn=cmd_research_ends)
    p = sub.add_parser("preregister")
    p.add_argument("family_id"); p.add_argument("config_hash"); p.set_defaults(fn=cmd_preregister)
    p = sub.add_parser("example")
    p.add_argument("--stage", default="importance",
                   choices=["bars", "labels", "features", "importance", "model"])
    p.add_argument("--symbol", default="SYNTH")
    p.set_defaults(fn=cmd_example)
    p = sub.add_parser("app")
    p.add_argument("--address", default="127.0.0.1",
                   help="bind address; use your VPN (e.g. Tailscale) IP, never a public one")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(fn=cmd_app)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)

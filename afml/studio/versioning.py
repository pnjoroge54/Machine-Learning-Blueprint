"""
Code version of a run: the git commit, the source hashes of the registered
callables the config uses, and a hash of the installed packages.

A scikit-learn or Numba upgrade can change results with no change to your code, so
the package hash is part of the version. The full environment (Python version, the
package list and, when available, ``conda env export``) is saved in each run folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from typing import Any, Dict, List, Tuple

from . import paths, registry
from .schemas import RunSpec, canonical_json


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=paths.repo_root(), capture_output=True,
                             text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_state() -> Dict[str, Any]:
    commit = _git("rev-parse", "HEAD") or "unknown"
    # Dirty tracked files in afml/ change results; untracked scratch files elsewhere do not
    diff = _git("diff", "HEAD", "--", "afml")
    return {
        "commit": commit,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
    }


@lru_cache(maxsize=1)
def installed_packages() -> Tuple[Tuple[str, str], ...]:
    pkgs = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower()
        if name:
            pkgs[name] = dist.version
    return tuple(sorted(pkgs.items()))


def packages_hash() -> str:
    return hashlib.sha256(json.dumps(installed_packages()).encode()).hexdigest()[:16]


def callables_used(spec: RunSpec) -> List[Tuple[str, str]]:
    spec = spec.resolved()
    used = [("source", spec.data.source)]
    if spec.labels is not None:
        if spec.labels.from_run is None:
            used.append(("strategy", spec.labels.strategy.name))
        used.append(("target", spec.labels.target.name))
    if spec.features is not None:
        used.extend(("feature", s.name) for s in spec.features.steps)
    if spec.importance is not None:
        used.append(("estimator", spec.importance.estimator.name))
    if spec.model is not None:
        used.append(("estimator", spec.model.estimator.name))
    return sorted(set(used))


@lru_cache(maxsize=1)
def studio_hash() -> str:
    """Hash of the studio core's own sources (untracked files never show in git diff)."""
    root = paths.repo_root() / "afml" / "studio"
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "tests" in f.parts:
            continue
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def code_version(spec: RunSpec) -> Dict[str, Any]:
    fn_versions = {f"{k}:{n}": registry.get(k, n).source_hash() for k, n in callables_used(spec)}
    version = {"git": git_state(), "studio": studio_hash(), "functions": fn_versions,
               "packages": packages_hash()}
    version["hash"] = hashlib.sha256(canonical_json(version).encode()).hexdigest()
    return version


def identity(config_hash: str, code_version_hash: str) -> str:
    """Identical identities must produce identical outputs."""
    return hashlib.sha256(f"{config_hash}:{code_version_hash}".encode()).hexdigest()


def environment_snapshot(conda_export: bool = True) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(installed_packages()),
        "conda_env_export": None,
    }
    if conda_export and not os.environ.get("AFML_STUDIO_SKIP_CONDA_EXPORT"):
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        if conda:
            try:
                out = subprocess.run([conda, "env", "export", "-p", prefix], capture_output=True,
                                     text=True, timeout=120)
                if out.returncode == 0:
                    snap["conda_env_export"] = out.stdout
            except (OSError, subprocess.TimeoutExpired):
                pass
    return snap

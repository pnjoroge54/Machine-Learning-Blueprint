"""
Utilities to lazily or eagerly warm-up @njit(cache=True) functions so their
compiled signatures are available in-process and on-disk. Provides:
 - register_numba_dummy: register tiny dummy args per-dispatcher name
 - lazy_warmup: wrapper to ensure compilation on first real call
 - prewarm_numba_in_package: eager scanner to compile many functions once
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numba
import numpy as np
from numba.core.registry import CPUDispatcher  # type: ignore

# Registry for function-name → (args_tuple, kwargs_dict)
_NUMBA_DUMMY_ARGS: Dict[str, Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}


def register_numba_dummy(
    fn_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any] | None = None
) -> None:
    """
    Register tiny dummy args to use when compiling a specific numba dispatcher.

    Args:
        fn_name: dispatcher.__name__ (string) used as key.
        args: positional args tuple for the dummy compile call.
        kwargs: optional kwargs dict for the dummy compile call.
    """
    _NUMBA_DUMMY_ARGS[fn_name] = (args, kwargs or {})


def _default_dummy_for_dispatcher() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Very small default dummy args commonly compatible with simple array-based signatures.
    Adjust or register explicit dummies for functions with different signatures.
    """
    return (np.array([np.int64(0)]), np.array([np.int64(0)]), np.array([np.int64(0)])), {}


def lazy_warmup(numba_dispatcher: CPUDispatcher):
    """
    Return a wrapper that ensures the given Numba dispatcher is compiled on first call.

    Usage:
        from afml.numba_warmup import lazy_warmup
        _precompute_active_indices_nopython = lazy_warmup(_precompute_active_indices_nopython)

    The wrapper:
      - If dispatcher.signatures is non-empty, calls dispatcher directly.
      - Otherwise, attempts a tiny dummy call (registered or default) to trigger
        compilation (and on-disk cache when @njit(cache=True)).
      - If the dummy invocation raises, it will still attempt the real call and
        surface the original error if any.
    """
    dispatcher = numba_dispatcher
    name = getattr(dispatcher, "__name__", repr(dispatcher))

    @functools.wraps(dispatcher)
    def wrapper(*args, **kwargs):
        # Already compiled in this process
        if dispatcher.signatures:
            return dispatcher(*args, **kwargs)

        # Find registered dummy or fallback
        dummy_info = _NUMBA_DUMMY_ARGS.get(name)
        if dummy_info is None:
            dummy_args, dummy_kwargs = _default_dummy_for_dispatcher()
        else:
            dummy_args, dummy_kwargs = dummy_info

        # Try to compile with dummy inputs; ignore compile-time failures gracefully
        try:
            dispatcher(*dummy_args, **dummy_kwargs)
        except Exception:
            # If dummy signature mismatches, skip the dummy compile attempt.
            # We still let the real call proceed, which may compile a matching signature.
            pass

        # Call the dispatcher for real (may compile another signature)
        return dispatcher(*args, **kwargs)

    # Expose introspection helpers
    wrapper._numba_dispatcher = dispatcher  # type: ignore
    wrapper._is_lazy_warmup = True  # type: ignore
    wrapper._dispatch_name = name  # type: ignore
    return wrapper


def _iter_modules_in_package(package_name: str):
    pkg = importlib.import_module(package_name)
    prefix = pkg.__name__ + "."
    for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix):
        yield name


def prewarm_numba_in_package(package_name: str = "afml", verbose: bool = False):
    """
    Discover Numba CPUDispatcher objects in package modules and attempt to warm them up.

    Behaviour:
      - Imports package modules under package_name.
      - For each attribute that looks like a Numba dispatcher, if it has no compiled
        signatures yet, calls it once with a registered dummy or the default dummy.
      - Returns list of (module_name, attr_name) that were successfully warmed.

    Notes:
      - Only warms functions for which the dummy signature matches. Register correct
        dummy args for functions with other signatures before calling this.
      - Prefer calling this inside each worker process initializer when using multiprocessing.
    """
    warmed = []
    for module_name in _iter_modules_in_package(package_name):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name)
            except Exception:
                continue
            # Heuristic: CPUDispatcher exposes `signatures` and `inspect_types`
            if not hasattr(attr, "signatures") or not hasattr(attr, "inspect_types"):
                continue
            dispatcher = attr
            # Skip already compiled in this process
            try:
                if dispatcher.signatures:
                    continue
            except Exception:
                # If introspection fails, skip
                continue

            name = getattr(dispatcher, "__name__", attr_name)
            dummy_info = _NUMBA_DUMMY_ARGS.get(name)
            if dummy_info is None:
                dummy_args, dummy_kwargs = _default_dummy_for_dispatcher()
            else:
                dummy_args, dummy_kwargs = dummy_info

            try:
                dispatcher(*dummy_args, **dummy_kwargs)
                warmed.append((module_name, attr_name))
            except Exception:
                # signature mismatch or other error; skip
                continue

    if verbose:
        print(f"Prewarmed {len(warmed)} numba functions: {warmed}")
    return warmed

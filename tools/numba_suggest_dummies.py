"""
Scan local afml package for Numba dispatchers and print their module + attribute.
Use this information to register accurate dummy args via register_numba_dummy.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

import afml


def find_dispatchers(package_name="afml"):
    pkg = importlib.import_module(package_name)
    prefix = pkg.__name__ + "."
    found = []
    for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name)
            except Exception:
                continue
            if hasattr(attr, "signatures") and hasattr(attr, "inspect_types"):
                try:
                    src = (
                        inspect.getsource(attr.py_func)
                        if hasattr(attr, "py_func")
                        else inspect.getsource(attr)
                    )
                except Exception:
                    src = "<source unavailable>"
                found.append((name, attr_name, src.splitlines()[:10]))
    return found


if __name__ == "__main__":
    for module_name, attr_name, src_head in find_dispatchers():
        print(f"{module_name}: {attr_name}")
        for ln in src_head:
            print("  ", ln)
        print("-" * 60)
        print("-" * 60)
        print("-" * 60)

"""
Conftest for new_api tests.

afml/__init__.py eagerly imports afml.production → tick_data_loader →
detect_distributed_frameworks(), which does `import dask.distributed`.
On this machine that import triggers a loguru background-thread access
violation on Windows.  We pre-populate sys.modules with MagicMock stubs
before any test file is imported so the real dask.distributed is never
loaded.

This file is processed by pytest BEFORE it imports any test module, so
the stubs are in place when afml/__init__.py first runs.
"""

import sys
from unittest.mock import MagicMock

for _mod_name in ("dask", "dask.distributed", "dask.base", "dask.dataframe"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

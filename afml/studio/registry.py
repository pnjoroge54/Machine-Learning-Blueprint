"""
Stable name -> callable registry.

Only registered callables can be selected from the studio, which keeps every
config serializable (a name and a params dict, never a live function) and every
run reproducible. Renaming a registered function is a breaking change: keep the
old name as an alias.

Parameter forms are generated from ``inspect.signature`` and type hints:
``Literal`` becomes a select box, ``bool`` a toggle, bounded numbers a number
input, and anything else a JSON field.
"""

from __future__ import annotations

import hashlib
import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

KINDS = ("source", "feature", "strategy", "target", "estimator")


@dataclass(frozen=True)
class Entry:
    kind: str
    name: str
    func: Callable
    description: str = ""
    aliases: Tuple[str, ...] = ()
    hidden: bool = False
    # Number of leading positional parameters supplied by the pipeline (e.g. ``bars``),
    # excluded from the generated form.
    n_inputs: int = 0

    @property
    def qualname(self) -> str:
        return f"{self.func.__module__}.{self.func.__qualname__}"

    def source_hash(self) -> str:
        """Hash of the callable's source, used in the run's code version."""
        try:
            src = inspect.getsource(self.func)
        except (OSError, TypeError):
            src = self.qualname
        return hashlib.sha256(src.encode()).hexdigest()[:16]


@dataclass
class ParamField:
    name: str
    widget: str  # "select" | "toggle" | "int" | "float" | "text" | "json"
    default: Any = None
    required: bool = False
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    annotation: str = ""
    help: str = ""


_REGISTRY: Dict[Tuple[str, str], Entry] = {}
_ALIASES: Dict[Tuple[str, str], str] = {}


class RegistryError(KeyError):
    pass


def register(
    kind: str,
    name: str,
    *,
    aliases: Tuple[str, ...] = (),
    description: str = "",
    hidden: bool = False,
    n_inputs: int = 0,
) -> Callable[[Callable], Callable]:
    """Decorator registering ``func`` under ``(kind, name)``."""
    if kind not in KINDS:
        raise ValueError(f"Unknown registry kind {kind!r}; expected one of {KINDS}")

    def decorator(func: Callable) -> Callable:
        key = (kind, name)
        existing = _REGISTRY.get(key)
        if existing is not None and existing.qualname != f"{func.__module__}.{func.__qualname__}":
            raise ValueError(f"{kind}:{name} is already registered to {existing.qualname}")
        doc = description or (inspect.getdoc(func) or "").split("\n\n")[0]
        _REGISTRY[key] = Entry(kind, name, func, doc, tuple(aliases), hidden, n_inputs)
        for alias in aliases:
            _ALIASES[(kind, alias)] = name
        return func

    return decorator


def get(kind: str, name: str) -> Entry:
    _ensure_loaded()
    resolved = _ALIASES.get((kind, name), name)
    try:
        return _REGISTRY[(kind, resolved)]
    except KeyError:
        raise RegistryError(f"No {kind} registered as {name!r}. Known: {names(kind)}") from None


def names(kind: str, include_hidden: bool = False) -> List[str]:
    _ensure_loaded()
    return sorted(n for (k, n), e in _REGISTRY.items() if k == kind and (include_hidden or not e.hidden))


def entries(kind: str, include_hidden: bool = False) -> List[Entry]:
    return [get(kind, n) for n in names(kind, include_hidden)]


def call(kind: str, name: str, *inputs, **params):
    """Resolve ``(kind, name)`` and call it with validated params."""
    entry = get(kind, name)
    unknown = set(params) - {f.name for f in param_fields(kind, name)}
    if unknown and not _accepts_var_kwargs(entry.func):
        raise TypeError(f"{kind}:{name} got unknown params {sorted(unknown)}")
    return entry.func(*inputs, **params)


def _accepts_var_kwargs(func: Callable) -> bool:
    return any(p.kind is p.VAR_KEYWORD for p in inspect.signature(func).parameters.values())


# --------------------------------------------------------------------------- forms


def param_fields(kind: str, name: str) -> List[ParamField]:
    """Describe the form fields for a registered callable's parameters."""
    entry = get(kind, name)
    sig = inspect.signature(entry.func)
    try:
        hints = typing.get_type_hints(entry.func, include_extras=True)
    except Exception:
        hints = {}

    fields: List[ParamField] = []
    params = [
        p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ][entry.n_inputs :]
    for p in params:
        ann = hints.get(p.name, p.annotation)
        default = None if p.default is inspect.Parameter.empty else p.default
        fields.append(_field_from_annotation(p.name, ann, default, p.default is inspect.Parameter.empty))
    return fields


def default_params(kind: str, name: str) -> Dict[str, Any]:
    return {f.name: f.default for f in param_fields(kind, name) if not f.required}


def _field_from_annotation(name: str, ann: Any, default: Any, required: bool) -> ParamField:
    lo = hi = None
    help_text = ""
    origin = typing.get_origin(ann)

    if origin is typing.Annotated:
        base, *meta = typing.get_args(ann)
        for m in meta:
            lo = getattr(m, "ge", getattr(m, "gt", lo))
            hi = getattr(m, "le", getattr(m, "lt", hi))
            if isinstance(m, str):
                help_text = m
        ann, origin = base, typing.get_origin(base)

    # Optional[X] -> X
    if origin is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            ann, origin = args[0], typing.get_origin(args[0])

    label = getattr(ann, "__name__", str(ann))
    if origin is typing.Literal:
        widget, choices = "select", list(typing.get_args(ann))
        return ParamField(name, widget, default, required, choices, annotation="Literal", help=help_text)
    if ann is bool:
        widget = "toggle"
    elif ann is int:
        widget = "int"
    elif ann is float:
        widget = "float"
    elif ann is str:
        widget = "text"
    else:
        widget = "json"
    return ParamField(name, widget, default, required, None, lo, hi, label, help_text)


# --------------------------------------------------------------------------- loading


_loaded = False


def _ensure_loaded() -> None:
    """Import the adapter module, whose import registers the built-in callables."""
    global _loaded
    if not _loaded:
        _loaded = True
        from .adapters import pipelines  # noqa: F401

#!/usr/bin/env python3
"""
Docstring normalizer: convert function/class/module docstrings to a uniform style.

Supports target styles: "google", "numpy", "sphinx".

Usage example:
    from docstring_normalizer import DocstringNormalizer
    dn = DocstringNormalizer(target_style="google")
    rewritten = dn.normalize_file("afml/some_module.py")
    if rewritten is not None:
        with open("afml/some_module.py", "w", encoding="utf-8") as f:
            f.write(rewritten)
"""

from __future__ import annotations

import ast
import io
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple


# ---------- Helpers: parse docstrings (best-effort) ----------

_PARAM_RE_SPHINX = re.compile(r":param\s+(?P<name>\w+)\s*:\s*(?P<desc>.*)")
_TYPE_RE_SPHINX = re.compile(r":type\s+(?P<name>\w+)\s*:\s*(?P<type>.+)")
_RTYPE_RE_SPHINX = re.compile(r":rtype\s*:\s*(?P<type>.+)")
_PARAM_GOOGLE_RE = re.compile(r"^\s*(?P<name>\w+)\s*\((?P<type>[^\)]+)\)\s*:\s*(?P<desc>.*)$")
_PARAM_NUMPY_RE = re.compile(r"^\s*(?P<name>\w+)\s*:\s*(?P<type>.+)$")
_RETURNS_GOOGLE_RE = re.compile(r"^\s*(?P<type>[^\:]+)\s*:\s*(?P<desc>.*)$")


def _split_sections(doc: str) -> List[str]:
    return doc.splitlines()


def parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, Dict[str, str]], Optional[Dict[str, str]]]:
    """
    Return (short_description, params, return_info)
    params: {name: {"type": type_str or "", "desc": description or ""}}
    return_info: {"type": type_str or "", "desc": description or ""} or None
    Best-effort across Sphinx, Google, NumPy styles.
    """
    if not doc:
        return "", {}, None

    lines = doc.expandtabs().splitlines()
    # Extract leading short description (first non-empty line)
    short = ""
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines):
        short = lines[i].strip()
        i += 1

    # join remaining for pattern searches
    remaining = "\n".join(lines[i:]) if i < len(lines) else ""

    params: Dict[str, Dict[str, str]] = {}
    return_info: Optional[Dict[str, str]] = None

    # Fast path: Sphinx-style :param: and :type:
    for m in _PARAM_RE_SPHINX.finditer(remaining):
        name = m.group("name")
        desc = m.group("desc").strip()
        params.setdefault(name, {})["desc"] = desc
    for m in _TYPE_RE_SPHINX.finditer(remaining):
        name = m.group("name")
        t = m.group("type").strip()
        params.setdefault(name, {})["type"] = t
    m_r = _RTYPE_RE_SPHINX.search(remaining)
    if m_r:
        return_info = {"type": m_r.group("type").strip(), "desc": ""}

    # Google-style Args / Returns blocks
    # Locate "Args:" or "Arguments:" block and parse lines underneath
    def _parse_block(header_names: List[str]) -> List[str]:
        out: List[str] = []
        for name in header_names:
            pat = re.compile(r"(?m)^\s*" + re.escape(name) + r"\s*:\s*$")
            m = pat.search(remaining)
            if m:
                start = m.end()
                # collect subsequent indented lines
                sub = remaining[start:]
                for line in sub.splitlines():
                    if line.strip() == "":
                        out.append("")
                        continue
                    if line.startswith(" ") or line.startswith("\t"):
                        out.append(line)
                        continue
                    break
                break
        return out

    args_block = _parse_block(["Args", "Arguments"])
    if args_block:
        # parse each non-empty line that looks like "name (type): desc"
        for raw in args_block:
            s = raw.strip()
            mg = _PARAM_GOOGLE_RE.match(s)
            if mg:
                name = mg.group("name")
                t = mg.group("type").strip()
                d = mg.group("desc").strip()
                params.setdefault(name, {})["type"] = t
                params.setdefault(name, {})["desc"] = d

    params_block = _parse_block(["Parameters"])
    if params_block:
        for raw in params_block:
            s = raw.strip()
            mn = _PARAM_NUMPY_RE.match(s)
            if mn:
                name = mn.group("name")
                t = mn.group("type").strip()
                params.setdefault(name, {})["type"] = t
                # description may follow in subsequent indented lines (not handled fully)

    # Google-style Returns
    returns_block = _parse_block(["Returns", "Return"])
    if returns_block:
        # take first non-empty line
        for raw in returns_block:
            s = raw.strip()
            if not s:
                continue
            mr = _RETURNS_GOOGLE_RE.match(s)
            if mr:
                return_info = {"type": mr.group("type").strip(), "desc": mr.group("desc").strip()}
            else:
                # fallback: single word type candidate
                first = s.split()[0]
                return_info = {"type": first, "desc": ""}
            break

    # If no structured info found, attempt to glean param: description lines from any "name : type" patterns in remaining
    if not params:
        for line in remaining.splitlines():
            mnp = _PARAM_NUMPY_RE.match(line.strip())
            if mnp:
                name = mnp.group("name")
                t = mnp.group("type").strip()
                params.setdefault(name, {})["type"] = t

    # Ensure every param has both keys
    for k, v in list(params.items()):
        params[k]["type"] = v.get("type", "")
        params[k]["desc"] = v.get("desc", "")

    return short, params, return_info


# ---------- Renderers ----------

def _render_google(short: str, params: Dict[str, Dict[str, str]], ret: Optional[Dict[str, str]], indent: str) -> str:
    parts: List[str] = []
    if short:
        parts.append(short)
        parts.append("")
    if params:
        parts.append("Args:")
        for name, info in params.items():
            typ = info.get("type", "").strip() or "Any"
            desc = info.get("desc", "").strip() or ""
            parts.append(f"    {name} ({typ}): {desc}")
        parts.append("")
    if ret:
        typ = ret.get("type", "").strip() or "Any"
        desc = ret.get("desc", "").strip() or ""
        parts.append("Returns:")
        parts.append(f"    {typ}: {desc}")
        parts.append("")
    # join with proper indentation
    body = "\n".join(parts).rstrip()
    if body:
        body = textwrap.indent(body, indent)
    return body


def _render_numpy(short: str, params: Dict[str, Dict[str, str]], ret: Optional[Dict[str, str]], indent: str) -> str:
    parts: List[str] = []
    if short:
        parts.append(short)
        parts.append("")
    if params:
        parts.append("Parameters")
        parts.append("----------")
        for name, info in params.items():
            typ = info.get("type", "").strip() or "Any"
            desc = info.get("desc", "").strip() or ""
            parts.append(f"{name} : {typ}")
            if desc:
                parts.extend(textwrap.wrap(desc, width=72))
        parts.append("")
    if ret:
        parts.append("Returns")
        parts.append("-------")
        typ = ret.get("type", "").strip() or "Any"
        desc = ret.get("desc", "").strip() or ""
        parts.append(f"{typ}")
        if desc:
            parts.append(desc)
        parts.append("")
    body = "\n".join(parts).rstrip()
    if body:
        body = textwrap.indent(body, indent)
    return body


def _render_sphinx(short: str, params: Dict[str, Dict[str, str]], ret: Optional[Dict[str, str]], indent: str) -> str:
    parts: List[str] = []
    if short:
        parts.append(short)
        parts.append("")
    for name, info in params.items():
        typ = info.get("type", "").strip()
        desc = info.get("desc", "").strip()
        parts.append(f":param {name}: {desc}")
        if typ:
            parts.append(f":type {name}: {typ}")
    if ret:
        typ = ret.get("type", "").strip()
        desc = ret.get("desc", "").strip()
        if desc:
            parts.append(f":return: {desc}")
        if typ:
            parts.append(f":rtype: {typ}")
    body = "\n".join(parts).rstrip()
    if body:
        body = textwrap.indent(body, indent)
    return body


_RENDERERS = {
    "google": _render_google,
    "numpy": _render_numpy,
    "sphinx": _render_sphinx,
}


# ---------- Normalizer class ----------

class DocstringNormalizer:
    def __init__(self, target_style: str = "google"):
        target_style = target_style.lower()
        if target_style not in _RENDERERS:
            raise ValueError(f"Unsupported style {target_style!r}; choose one of {list(_RENDERERS)}")
        self.target_style = target_style

    def normalize_file(self, path: str, write: bool = False) -> Optional[str]:
        """
        Read file `path`, normalize docstrings for module, top-level functions, classes, and methods.
        If write=True, overwrite the file. Returns rewritten source or None if no change.
        """
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
        new_src = self._normalize_source(src, tree)
        if new_src is None:
            return None
        if write:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
        return new_src

    def _normalize_source(self, src: str, tree: ast.Module) -> Optional[str]:
        """
        Return modified source text or None if no changes were necessary.
        Strategy:
        - For each node that can have a docstring (Module, FunctionDef, AsyncFunctionDef, ClassDef),
          extract the docstring and its location, build a normalized docstring, and replace it in source.
        - Replacement preserves triple-quote style (''' or """) when possible and indentation.
        """
        # Collect edits as (start_idx, end_idx, replacement_text)
        edits: List[Tuple[int, int, str]] = []

        # Helper to get node docstring range
        for node in [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]:
            # Only consider top-level FunctionDef/ClassDef or nested ones too (methods)
            doc = ast.get_docstring(node, clean=False)
            if doc is None:
                continue
            # find the Expr node that holds the docstring
            if not node.body:
                continue
            first = node.body[0]
            if not isinstance(first, ast.Expr):
                continue
            # The value should be a Constant or Str node in ast
            value_node = first.value
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                # Compute exact source slice by using lineno/col_offset and end_lineno/end_col_offset (python 3.8+)
                if not hasattr(value_node, "lineno") or not hasattr(value_node, "end_lineno"):
                    continue
                start_line = value_node.lineno
                start_col = value_node.col_offset
                end_line = value_node.end_lineno
                end_col = value_node.end_col_offset
                src_lines = src.splitlines(keepends=True)
                # compute absolute indices
                start_idx = sum(len(l) for l in src_lines[: start_line - 1]) + start_col
                end_idx = sum(len(l) for l in src_lines[: end_line - 1]) + end_col
                # Determine indentation for docstring content (based on node.col_offset)
                indent = " " * (node.col_offset + 4) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else " " * node.col_offset
                # Parse old docstring into structured pieces
                short, params, ret = parse_docstring(doc)
                # Render new docstring body
                renderer = _RENDERERS[self.target_style]
                body = renderer(short, params, ret, indent="")
                # Build full triple-quoted string with same quote characters as original
                # Inspect original raw docstring text to see preferred quoting
                raw_doc_text = src[start_idx:end_idx]
                quote_match = re.match(r'\s*([uU]{0,1}[rR]{0,1})(?P<quote>["\']{3})', raw_doc_text)
                if quote_match:
                    quote_chars = quote_match.group("quote")
                else:
                    # fallback to triple double-quotes
                    quote_chars = '"""'
                # Build final docstring text with one blank line after short description if present
                # Indent body lines to match node indentation
                if body:
                    # indent body by node indentation + 4 spaces for function content
                    target_indent = " " * (node.col_offset + 4) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else " " * node.col_offset
                    # ensure body lines are indented, but keep leading blank lines minimal
                    body_lines = body.splitlines()
                    body_indented = "\n".join((target_indent + line) if line.strip() != "" else "" for line in body_lines)
                    new_inner = "\n" + body_indented + ("\n" + target_indent if not body_indented.endswith("\n") else "")
                else:
                    new_inner = "\n" + (" " * (node.col_offset + 4)) + "\n"

                replacement = quote_chars + new_inner + quote_chars
                edits.append((start_idx, end_idx, replacement))

        if not edits:
            return None

        # Apply edits in reverse order to avoid index invalidation
        new_src = src
        for s, e, rep in sorted(edits, key=lambda x: x[0], reverse=True):
            new_src = new_src[:s] + rep + new_src[e:]

        return new_src


# ---------- CLI utility to process directory ----------

def normalize_directory(root: str, target_style: str = "google", write: bool = False) -> List[Tuple[str, bool]]:
    """
    Walk `root` and normalize all .py files. Returns list of (path, changed_bool).
    If write=True, files are overwritten.
    """
    dn = DocstringNormalizer(target_style=target_style)
    changed: List[Tuple[str, bool]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden directories
        if any(part.startswith(".") for part in dirpath.split(os.sep)):
            # still descend; change behavior if you want to skip
            pass
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                new = dn.normalize_file(path, write=write)
            except Exception:
                # conservative: skip files that fail parsing
                new = None
            changed.append((path, new is not None))
    return changed


# ---------- Example runner when invoked directly ----------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Normalize docstrings to a uniform style.")
    p.add_argument("root", help="Path to package or folder (e.g., ./afml)")
    p.add_argument("--style", choices=list(_RENDERERS.keys()), default="google")
    p.add_argument("--write", action="store_true", help="Overwrite files with normalized docstrings")
    args = p.parse_args()
    results = normalize_directory(args.root, target_style=args.style, write=args.write)
    for path, changed in results:
        print(f"{path}: {'changed' if changed else 'unchanged'}")
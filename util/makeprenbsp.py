#!/usr/bin/env python3
"""
make_pre_nbsp.py

Usage:
    python make_pre_nbsp.py input.html output.html

Replaces regular space characters inside <pre>...</pre> (and nested <code> tags)
with the HTML entity &nbsp; so editors that collapse spaces preserve formatting.
"""

import sys
import re

def replace_spaces_in_pre(html):
    # Pattern to capture <pre ...>...</pre> (non-greedy)
    pre_pattern = re.compile(r'(<pre\b[^>]*>)(.*?)(</pre>)', re.IGNORECASE | re.DOTALL)

    def repl(match):
        open_tag, content, close_tag = match.group(1), match.group(2), match.group(3)
        # Replace literal spaces with &nbsp; but leave existing &nbsp; intact.
        # To avoid double-encoding existing entities, we first temporarily mask them.
        mask = '__AMP_NBSP_MASK__'
        content = content.replace('&nbsp;', mask)
        # Replace all regular space characters with &nbsp;
        content = content.replace(' ', '&nbsp;')
        # Restore masked entities (if any)
        content = content.replace(mask, '&nbsp;')
        return f"{open_tag}{content}{close_tag}"

    return pre_pattern.sub(repl, html)

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_pre_nbsp.py input.html output.html")
        sys.exit(2)

    infile, outfile = sys.argv[1], sys.argv[2]

    with open(infile, 'r', encoding='utf-8') as f:
        html = f.read()

    new_html = replace_spaces_in_pre(html)

    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(new_html)

    print(f"Wrote {outfile} (pre blocks converted).")

if __name__ == '__main__':
    main()
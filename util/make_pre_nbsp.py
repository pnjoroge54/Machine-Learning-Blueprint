#!/usr/bin/env python3
"""
make_pre_nbsp.py

Usage:
    python make_pre_nbsp.py input.html output.html

Converts various code block formats to simple format with &nbsp; spacing.
Handles split <code> blocks that span multiple <p> tags.
"""

import re
import sys
from html import unescape


def merge_split_code_blocks(html):
    """
    Merge <code> blocks that have been split across <p> tags.
    Converts: <code>line1</code></p><p><code>line2</code>
    To: <code>line1\nline2</code>
    """
    # Pattern to find </code></p><p><code> sequences
    split_pattern = re.compile(r"</code>\s*</p>\s*<p>\s*<code>", re.IGNORECASE)

    # Replace with newline to rejoin the content
    return split_pattern.sub("\n", html)


def wrap_adjacent_code_in_pre(html):
    """
    Wrap adjacent <code> blocks (that should be in same pre) with <pre class="text">.
    Looks for patterns like: <p><code>...</code></p><p><code>...</code></p>
    """
    # First merge the split code blocks
    html = merge_split_code_blocks(html)

    # Pattern to find standalone code blocks in paragraphs
    # (code blocks that contain box-drawing characters or ASCII art)
    code_block_pattern = re.compile(
        r"<p>\s*<code>([^<]*[╔║╠╚═─┌┐└┘│├┤┬┴┼▼▲►◄][^<]*)</code>\s*</p>",
        re.IGNORECASE | re.DOTALL,
    )

    def has_adjacent_code_blocks(match, html):
        """Check if this code block has adjacent code blocks (is part of ASCII art)"""
        content = match.group(1)
        # Check if content has box-drawing chars
        box_chars = "╔║╠╚═─┌┐└┘│├┤┬┴┼▼▲►◄"
        return any(char in content for char in box_chars)

    # Find all code blocks that look like ASCII art
    matches = list(code_block_pattern.finditer(html))

    if not matches:
        return html

    # Group adjacent matches
    groups = []
    current_group = [matches[0]]

    for i in range(1, len(matches)):
        # Check if matches are adjacent (within 50 chars)
        if matches[i].start() - current_group[-1].end() < 50:
            current_group.append(matches[i])
        else:
            groups.append(current_group)
            current_group = [matches[i]]

    groups.append(current_group)

    # Process groups in reverse to maintain string positions
    for group in reversed(groups):
        if len(group) < 2:  # Only process groups of 2+ blocks
            continue

        # Extract all content
        combined_content = []
        for match in group:
            combined_content.append(match.group(1))

        # Create replacement
        full_content = "\n".join(combined_content)
        replacement = f'<pre class="text"><code>{full_content}</code></pre>'

        # Replace entire span
        start = group[0].start()
        end = group[-1].end()
        html = html[:start] + replacement + html[end:]

    return html


def simplify_pandoc_code_block(html):
    """Convert Pandoc's complex code blocks to simple format with syntax highlighting."""
    pandoc_pattern = re.compile(
        r'<div\s+class="sourceCode"[^>]*>.*?'
        r'<pre\s+class="sourceCode[^"]*"[^>]*>'
        r'<code\s+class="sourceCode[^"]*"[^>]*>(.*?)</code>'
        r"</pre>.*?</div>",
        re.IGNORECASE | re.DOTALL,
    )

    def process_pandoc_block(match):
        code_content = match.group(1)

        # Remove line number spans
        code_content = re.sub(
            r'<span\s+id="[^"]*"><a\s+href="[^"]*"[^>]*></a>(.*?)</span>',
            r"\1",
            code_content,
            flags=re.DOTALL,
        )

        # Simplify class names
        class_map = {
            "co": "comment",
            "kw": "keyword",
            "cf": "keyword",
            "op": "operator",
            "bu": "builtin",
            "st": "string",
            "dv": "number",
            "va": "variable",
        }

        for pandoc_class, simple_class in class_map.items():
            code_content = re.sub(
                rf'<span\s+class="{pandoc_class}">',
                f'<span class="{simple_class}">',
                code_content,
            )

        code_content = unescape(code_content)
        code_content = replace_spaces_outside_tags(code_content)
        code_content = re.sub(r"\n\n+", "\n", code_content)

        return f'<pre class="code">{code_content}</pre>'

    return pandoc_pattern.sub(process_pandoc_block, html)


def simplify_text_code_block(html):
    """Convert <pre class="text"><code>...</code></pre> to <pre class="text">...</pre>"""
    text_pattern = re.compile(
        r'<pre\s+class="text"[^>]*>\s*<code>(.*?)</code>\s*</pre>',
        re.IGNORECASE | re.DOTALL,
    )

    def process_text_block(match):
        content = match.group(1)
        content = unescape(content)
        content = content.replace(" ", "&nbsp;")
        return f'<pre class="text">{content}</pre>'

    return text_pattern.sub(process_text_block, html)


def replace_spaces_outside_tags(text):
    """Replace spaces outside HTML tags with &nbsp;"""
    result = []
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
            result.append(char)
        elif char == ">":
            in_tag = False
            result.append(char)
        elif char == " " and not in_tag:
            result.append("&nbsp;")
        else:
            result.append(char)
    return "".join(result)


def replace_spaces_in_remaining_pre(html):
    """Handle any remaining <pre> blocks"""
    pre_pattern = re.compile(r"(<pre\b[^>]*>)(.*?)(</pre>)", re.IGNORECASE | re.DOTALL)

    def repl(match):
        open_tag, content, close_tag = match.group(1), match.group(2), match.group(3)

        # Skip if already processed
        if "&nbsp;" in content:
            return match.group(0)

        # Check if it has nested <code> tags
        code_pattern = re.compile(r"^<code>(.*)</code>$", re.DOTALL)
        code_match = code_pattern.match(content.strip())

        if code_match:
            inner_content = code_match.group(1)
            inner_content = unescape(inner_content)
            inner_content = inner_content.replace(" ", "&nbsp;")

            class_match = re.search(r'class="([^"]+)"', open_tag)
            pre_class = class_match.group(1) if class_match else "code"

            return f'<pre class="{pre_class}">{inner_content}</pre>'

        content = unescape(content)
        content = replace_spaces_outside_tags(content)

        return f"{open_tag}{content}{close_tag}"

    return pre_pattern.sub(repl, html)


def main():
    if len(sys.argv) != 3:
        print("Usage: python make_pre_nbsp.py input.html output.html")
        sys.exit(2)

    infile, outfile = sys.argv[1], sys.argv[2]

    with open(infile, "r", encoding="utf-8") as f:
        html = f.read()

    # Process in order:
    # 1. Merge split code blocks and wrap in pre
    html = wrap_adjacent_code_in_pre(html)

    # 2. Pandoc code blocks with syntax highlighting
    html = simplify_pandoc_code_block(html)

    # 3. Text/ASCII art blocks
    html = simplify_text_code_block(html)

    # 4. Any remaining pre blocks
    html = replace_spaces_in_remaining_pre(html)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {outfile} (code blocks converted).")


if __name__ == "__main__":
    main()

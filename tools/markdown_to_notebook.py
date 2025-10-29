import os
import re

import nbformat as nbf


def markdown_to_notebook(markdown_content, output_filename="AFML Experiments.ipynb"):
    """
    Convert markdown content (either as string or file path) to Jupyter notebook.

    Args:
        markdown_content (str): Can be either markdown content string or file path
        output_filename (str): Output notebook filename

    Returns:
        str: Output notebook filename
    """
    # Read content if input is a valid file path
    if isinstance(markdown_content, str) and os.path.isfile(markdown_content):
        with open(markdown_content, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = markdown_content

    # Create new notebook and pattern for code blocks
    nb = nbf.v4.new_notebook()
    pattern = r"```python\n(.*?)\n```"
    current_pos = 0
    cells = []

    # Process all code blocks
    for match in re.finditer(pattern, content, re.DOTALL):
        # Add preceding markdown
        markdown_segment = content[current_pos : match.start()].strip()
        if markdown_segment:
            clean_md = re.sub(r"\n{3,}", "\n\n", markdown_segment)
            cells.append(nbf.v4.new_markdown_cell(clean_md))

        # Add code block
        code_block = match.group(1).strip()
        if code_block:
            cells.append(nbf.v4.new_code_cell(code_block))

        current_pos = match.end()

    # Add remaining markdown after last code block
    trailing_markdown = content[current_pos:].strip()
    if trailing_markdown:
        clean_trailing = re.sub(r"\n{3,}", "\n\n", trailing_markdown)
        cells.append(nbf.v4.new_markdown_cell(clean_trailing))

    # Handle case with no code blocks
    if not cells:
        clean_content = re.sub(r"\n{3,}", "\n\n", content.strip())
        cells.append(nbf.v4.new_markdown_cell(clean_content))

    # Add cells to notebook and save
    nb.cells = cells
    with open(output_filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook saved as {output_filename}")
    return output_filename

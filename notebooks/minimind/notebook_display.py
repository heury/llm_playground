"""Display experiment rows as Markdown, retaining the interactive table on demand."""

import html
import re

import marimo as mo


def markdown_table(rows):
    """Build a Markdown table from ordered dictionaries, escaping cell contents."""
    rows = list(rows)
    if not rows:
        return "*표시할 행이 없습니다.*"
    columns = list(dict.fromkeys(key for row in rows for key in row))

    def cell(value):
        text = html.escape("" if value is None else str(value), quote=False)
        text = re.sub(r"([\\`*_$|])", r"\\\1", text)
        # marimo treats \[...\] as display math, even inside table cells.
        # Entities preserve literal brackets without creating math delimiters.
        text = text.replace("[", "&#91;").replace("]", "&#93;")
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

    lines = ["| " + " | ".join(cell(key) for key in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    lines.extend("| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows)
    return "\n".join(lines)


def result_table(rows, **table_options):
    """Show Markdown first; keep sorting, searching and pagination in a closed panel."""
    rows = list(rows)
    if not rows:
        return mo.md(markdown_table(rows))
    return mo.vstack([
        mo.md(markdown_table(rows)),
        mo.accordion({"정렬·검색·페이지 보기": mo.ui.table(rows, **table_options)}),
    ])

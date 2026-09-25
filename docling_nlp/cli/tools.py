#!/usr/bin/env python
"""CLI with helper tools for docling-nlp"""

import sys

try:
    import typer
except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else "typer"
    print(
        f"Error: Missing required CLI dependency '{missing_package}'", file=sys.stderr
    )
    print(
        "\nThe docling-nlp-tools CLI requires additional dependencies.", file=sys.stderr
    )
    print("Please install them using one of the following options:\n", file=sys.stderr)
    print("  1. Install the full docling-nlp package (recommended):", file=sys.stderr)
    print("     pip install docling-nlp\n", file=sys.stderr)
    print("  2. Install just the missing dependencies:", file=sys.stderr)
    print("     pip install typer rich\n", file=sys.stderr)
    sys.exit(1)

from docling_nlp.cli.models import app as models_app

app = typer.Typer(
    name="Docling NLP helpers",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

app.add_typer(models_app, name="models")

click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()

#!/usr/bin/env python
"""CLI to download and inspect the pretrained NLP model artifacts"""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, List, Optional

try:
    import typer
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
except ImportError as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else "typer or rich"
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

from docling_nlp.utils.load_pretrained_models import (
    copy_support_resources,
    download_pretrained_nlp_models,
    get_resources_dir,
    list_pretrained_nlp_models,
    load_models_config,
    locked_model_revision,
    read_models_lock,
)
from docling_nlp.utils.model_info import (
    MODEL_INFO_SUFFIX,
    NlpModelInfo,
    describe_nlp_model,
    format_size,
)

_log = logging.getLogger(__name__)

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="Docling NLP models helper",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _setup_logging(quiet: bool):
    if quiet:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="[blue]%(message)s[/blue]",
        datefmt="[%X]",
        handlers=[RichHandler(show_level=False, show_time=False, markup=True)],
    )


@app.command("list")
def list_models(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            "-o",
            "--output-dir",
            help="The directory the models are looked up in "
            "(default: the docling-nlp resources directory).",
        ),
    ] = None,
):
    """List the available NLP models and whether they are downloaded."""

    resources_dir = Path(get_resources_dir())
    target_dir = Path(output_dir) if output_dir is not None else resources_dir

    specs = list_pretrained_nlp_models(resources_dir)
    huggingface = load_models_config(resources_dir)["huggingface"]
    repo_id = huggingface["repo-id"]
    revision = huggingface.get("revision")
    lock = read_models_lock(target_dir)

    table = Table(title=f"NLP models of {repo_id}")
    table.add_column("name", style="bold")
    table.add_column("kind")
    table.add_column("path", overflow="fold")
    table.add_column("downloaded")

    stale = []
    for spec in specs:
        downloaded = spec.target(target_dir).exists()
        local_revision = locked_model_revision(lock, spec.name, repo_id)

        if not downloaded:
            status = "[red]no[/red]"
        elif local_revision == revision:
            status = "[green]yes[/green]"
        else:
            status = "[yellow]stale[/yellow]"
            stale.append(spec.name)

        table.add_row(spec.name, spec.kind, spec.relative_path, status)

    console.print(table)
    console.print(f"Models directory: {target_dir}")
    if revision:
        console.print(f"Revision: {revision}")

    if stale:
        console.print(
            f"\n[yellow]{len(stale)} model(s) are not from revision "
            f"{(revision or 'main')[:8]}[/yellow], or were downloaded before the "
            f"revision was recorded. Run `docling-nlp-tools models download "
            f"{' '.join(stale)}` to update them."
        )


@app.command("download")
def download(
    models: Annotated[
        List[str] | None,
        typer.Argument(
            help="Models to download (default behavior: all models are downloaded). "
            "Run `docling-nlp-tools models list` for the available names.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            "-o",
            "--output-dir",
            help="The directory where to download the models "
            "(default: the docling-nlp resources directory).",
        ),
    ] = None,
    kind: Annotated[
        List[str] | None,
        typer.Option(
            ...,
            "-k",
            "--kind",
            help="Download all models of a kind, e.g. 'crf', 'fasttext' or 'rgx'. "
            "Repeat for multiple (mutually exclusive with naming models).",
        ),
    ] = None,
    all: Annotated[
        bool,
        typer.Option(
            ...,
            "--all",
            help="If true, all available models will be downloaded "
            "(mutually exclusive with passing specific models).",
            show_default=True,
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option(..., help="If true, the download will be forced.")
    ] = False,
    standalone: Annotated[
        bool,
        typer.Option(
            ...,
            "--standalone/--no-standalone",
            help="If true (default), the packaged support resources (confusables, "
            "regex data, JSON configurations) are copied next to the models, so that "
            "the output directory can be used as DOCLING_NLP_RESOURCES_DIR. Only "
            "applies when --output-dir is set.",
        ),
    ] = True,
    quiet: Annotated[
        bool,
        typer.Option(
            ...,
            "-q",
            "--quiet",
            help="No extra output is generated, the CLI prints only the directory "
            "with the downloaded models.",
        ),
    ] = False,
):
    """Download all (or a selection of) the pretrained NLP models."""

    if models and all:
        raise typer.BadParameter(
            "Cannot simultaneously set 'all' parameter and specify models to download."
        )
    if models and kind:
        raise typer.BadParameter(
            "Cannot simultaneously set 'kind' parameter and specify models to download.",
            param_hint="--kind",
        )
    if kind and all:
        raise typer.BadParameter(
            "Cannot simultaneously set 'all' and 'kind' parameters.",
            param_hint="--kind",
        )

    _setup_logging(quiet)

    resources_dir = Path(get_resources_dir())
    target_dir = Path(output_dir) if output_dir is not None else resources_dir

    specs = list_pretrained_nlp_models(resources_dir)

    if kind:
        known_kinds = sorted({spec.kind for spec in specs})
        unknown = sorted(set(kind) - set(known_kinds))
        if unknown:
            raise typer.BadParameter(
                f"Unknown model kind(s): {', '.join(unknown)}. "
                f"Available: {', '.join(known_kinds)}",
                param_hint="--kind",
            )
        to_download = [spec.name for spec in specs if spec.kind in kind]
    elif models:
        to_download = list(models)
    else:
        to_download = None  # all of them

    try:
        downloaded = download_pretrained_nlp_models(
            names=to_download,
            output_dir=target_dir,
            force=force,
            verbose=(not quiet),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    if standalone and target_dir != resources_dir:
        copy_support_resources(
            output_dir=target_dir,
            resources_dir=resources_dir,
            force=force,
            verbose=(not quiet),
        )

    if quiet:
        typer.echo(target_dir)
        return

    typer.secho(
        f"\nDownloaded {len(downloaded)} model(s) into: {target_dir}.", fg="green"
    )

    if target_dir != resources_dir:
        console.print(
            "\n",
            "This is not the default resources directory. To let docling-nlp pick up",
            "these models, point the resources directory to it:\n\n",
            f"`export DOCLING_NLP_RESOURCES_DIR={target_dir}`",
        )
        if not standalone:
            console.print(
                "\n",
                "[yellow]Note:[/yellow] --no-standalone was passed, so the support",
                "resources the NLP models also read (confusables, regex data, JSON",
                "configurations) were not copied along.",
            )


def _print_model_info(info: NlpModelInfo):
    """Render the model info as a set of rich tables."""

    console.print(Panel(info.summary or "(no description available)", title=info.name))

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")

    table.add_row("name", info.name)
    table.add_row("kind", info.kind)
    if info.task:
        table.add_row("task", info.task)
    if info.applies_to:
        table.add_row("applies to", ", ".join(info.applies_to))
    if info.dependencies:
        table.add_row("depends on", ", ".join(info.dependencies))

    if info.downloaded:
        table.add_row("size", format_size(info.size_bytes))
    elif info.remote_size_bytes is not None:
        table.add_row("size", f"{format_size(info.remote_size_bytes)} (on HuggingFace)")

    for key, value in info.details.items():
        table.add_row(key, str(value))

    if info.upstream:
        table.add_row("upstream", info.upstream)

    table.add_row("repository", info.repo_id)
    if info.revision:
        table.add_row("revision", info.revision)
    if info.downloaded:
        table.add_row(
            "local revision",
            info.local_revision
            or "[yellow]unknown, downloaded before the revision was recorded[/yellow]",
        )
    table.add_row("artifact", info.relative_path)
    table.add_row("path", str(info.path))
    table.add_row(
        "downloaded",
        "[green]yes[/green]" if info.downloaded else "[red]no[/red]",
    )
    if info.stale:
        table.add_row(
            "status",
            f"[yellow]stale, run `docling-nlp-tools models download {info.name}`"
            f"[/yellow]",
        )
    if info.info_path is not None:
        table.add_row("info file", str(info.info_path))

    console.print(table)

    heading = "labels"
    if info.label_description:
        heading = f"labels ({info.label_description})"

    if not info.labels:
        if info.downloaded:
            console.print(
                f"[bold]{heading}[/bold]: none could be read from the artifact"
            )
        else:
            console.print(
                f"[bold]{heading}[/bold]: unknown, the artifact is not downloaded. "
                f"Run `docling-nlp-tools models download {info.name}`."
            )
        return

    described = any(label.description for label in info.labels)
    counted = any(label.count is not None for label in info.labels)

    labels = Table(title=f"{heading}: {len(info.labels)}", box=box.SIMPLE_HEAD)
    labels.add_column("label", style="bold")
    if described:
        labels.add_column("description", overflow="fold")
    if counted:
        labels.add_column(info.label_count_header, justify="right")

    for label in sorted(info.labels, key=lambda label: label.name):
        row = [label.name]
        if described:
            row.append(label.description or "-")
        if counted:
            row.append(f"{label.count:,}" if label.count is not None else "-")
        labels.add_row(*row)

    console.print(labels)

    if not info.downloaded:
        console.print(
            f"[yellow]Note:[/yellow] the artifact is not downloaded, so these are the "
            f"labels its description declares. Run "
            f"`docling-nlp-tools models download {info.name}` to read them back from "
            f"the model itself."
        )
    elif not described:
        console.print(
            f"[yellow]Note:[/yellow] no label descriptions were found next to the "
            f"artifact. They are read from an optional "
            f"`{Path(info.relative_path).stem}{MODEL_INFO_SUFFIX}` file."
        )
    elif not counted or any(label.count is None for label in info.labels):
        console.print(
            "[yellow]Note:[/yellow] the labels without a count are described but do "
            "not occur in this artifact."
        )


@app.command("info")
def info(
    model: Annotated[
        str,
        typer.Argument(
            help="The model to describe. Run `docling-nlp-tools models list` "
            "for the available names.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            ...,
            "-o",
            "--output-dir",
            help="The directory the model artifact is looked up in "
            "(default: the docling-nlp resources directory).",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            ...,
            "--json",
            help="Print the model info as JSON, for scripting.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            ...,
            "--offline",
            help="Do not contact HuggingFace to look up the size of an artifact "
            "that is not downloaded yet.",
        ),
    ] = False,
):
    """Describe an NLP model: what it does, how big it is and which labels it produces."""

    resources_dir = Path(get_resources_dir())
    target_dir = Path(output_dir) if output_dir is not None else resources_dir

    try:
        info = describe_nlp_model(
            model,
            resources_dir=resources_dir,
            models_dir=target_dir,
            remote=(not offline),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    if as_json:
        typer.echo(json.dumps(info.to_dict(), indent=2))
        return

    _print_model_info(info)


click_app = typer.main.get_command(app)

if __name__ == "__main__":
    app()

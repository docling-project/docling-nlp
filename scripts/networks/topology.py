#!/usr/bin/env python
"""Summarize node and edge topology from annotated DocLang archives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from docling_nlp.andromeda_doclang import DocLangXDocument
from tabulate import tabulate

DEFAULT_MODELS = "language;term"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List node and edge types found in one or more .dclx files, "
            "or in folders containing .dclx files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .dclx files or folders containing .dclx files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input folders recursively for .dclx files.",
    )
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "NLP models used when a document has no annotations. "
            f"Default: {DEFAULT_MODELS}"
        ),
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Fail on unannotated documents instead of applying NLP models.",
    )
    parser.add_argument(
        "--write-annotated",
        action="store_true",
        help="Persist lazily annotated/materialized documents back to their input .dclx paths.",
    )
    parser.add_argument(
        "--no-materialize-edges",
        action="store_true",
        help="Do not materialize DocLang graph edges when a document has none.",
    )
    parser.add_argument(
        "--derived-entities",
        choices=("terms", "all", "none"),
        default="terms",
        help="Derived entities included when materializing graph edges. Default: terms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the topology summary as JSON to this path.",
    )
    return parser.parse_args()


def iter_dclx_files(inputs: list[Path], recursive: bool) -> list[Path]:
    paths: list[Path] = []

    for input_path in inputs:
        if input_path.is_dir():
            globber = input_path.rglob if recursive else input_path.glob
            paths.extend(sorted(globber("*.dclx")))
            continue

        if input_path.suffix.lower() != ".dclx":
            raise ValueError(f"not a .dclx file or directory: {input_path}")
        paths.append(input_path)

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError("no .dclx files found")

    return unique_paths


def read_document(
    path: Path,
    models: str,
    annotate_missing: bool,
    write_annotated: bool,
    materialize_edges: bool,
    derived_entities: str,
) -> DocLangXDocument:
    doc = DocLangXDocument()
    if not doc.read(str(path)):
        raise RuntimeError(f"could not read {path}: {doc.last_error()}")

    if not doc.has_annotations():
        if not annotate_missing:
            raise RuntimeError(f"{path} has no annotations")
        if not doc.apply_nlp(models, progress_every=0):
            raise RuntimeError(f"could not annotate {path}: {doc.last_error()}")
        if write_annotated and not doc.write(str(path)):
            raise RuntimeError(f"could not write annotated {path}: {doc.last_error()}")

    if materialize_edges and doc.edges().empty:
        doc.materialize_edges(derived_entities)
        if write_annotated and not doc.write(str(path)):
            raise RuntimeError(
                f"could not write materialized {path}: {doc.last_error()}"
            )

    return doc


def value_or_empty(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def add_dataframe_counts(
    counter: Counter[tuple[str, ...]],
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    if dataframe.empty:
        return

    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"missing expected columns: {', '.join(missing)}")

    for row in dataframe.loc[:, columns].itertuples(index=False):
        counter[tuple(value_or_empty(value) for value in row)] += 1


def add_entity_counts(
    node_counter: Counter[tuple[str, str, str]],
    mention_counter: Counter[tuple[str, str, str]],
    entities: pd.DataFrame,
) -> None:
    if entities.empty:
        return

    for row in entities.itertuples(index=False):
        key = (
            value_or_empty(row.type),
            value_or_empty(row.subtype),
            value_or_empty(row.entity_kind),
        )
        node_counter[key] += 1
        mention_counter[key] += int(row.count)


def filter_entities(entities: pd.DataFrame, derived_mode: str) -> pd.DataFrame:
    if entities.empty or derived_mode == "all":
        return entities
    if derived_mode == "none":
        return entities[entities["entity_kind"] == "exact"].copy()
    return entities[
        (entities["entity_kind"] == "exact")
        | ((entities["type"] == "term") & (entities["entity_kind"] == "derived"))
    ].copy()


def counter_rows(
    counter: Counter[tuple[str, ...]],
    columns: tuple[str, ...],
    count_column: str = "count",
) -> list[dict[str, Any]]:
    rows = [
        {**dict(zip(columns, key)), count_column: count}
        for key, count in counter.items()
    ]
    return sorted(rows, key=lambda row: tuple(str(row[column]) for column in columns))


def build_topology(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    node_counter: Counter[tuple[str, str, str]] = Counter()
    mention_counter: Counter[tuple[str, str, str]] = Counter()
    instance_counter: Counter[tuple[str, str]] = Counter()
    relation_counter: Counter[tuple[str]] = Counter()
    property_counter: Counter[tuple[str, str]] = Counter()

    for path in paths:
        doc = read_document(
            path,
            args.models,
            annotate_missing=not args.no_annotate,
            write_annotated=args.write_annotated,
            materialize_edges=not args.no_materialize_edges,
            derived_entities=args.derived_entities,
        )
        summary = doc.overview()
        documents.append({"path": str(path), **summary})

        add_entity_counts(
            node_counter,
            mention_counter,
            filter_entities(doc.entities(), args.derived_entities),
        )
        add_dataframe_counts(instance_counter, doc.instances(), ("type", "subtype"))
        add_dataframe_counts(relation_counter, doc.edges(), ("name",))
        add_dataframe_counts(property_counter, doc.properties(), ("type", "label"))

    node_rows = counter_rows(
        node_counter,
        ("type", "subtype", "entity_kind"),
        count_column="nodes",
    )
    for row in node_rows:
        key = (row["type"], row["subtype"], row["entity_kind"])
        row["mentions"] = mention_counter[key]

    return {
        "documents": documents,
        "nodes": node_rows,
        "instances": counter_rows(instance_counter, ("type", "subtype")),
        "edges": counter_rows(relation_counter, ("name",)),
        "properties": counter_rows(property_counter, ("type", "label")),
    }


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if rows:
        print(tabulate(rows, headers="keys", tablefmt="github"))
    else:
        print("(none)")


def print_topology(topology: dict[str, Any]) -> None:
    doc_rows = [
        {
            "path": row["path"],
            "properties": row["properties"],
            "instances": row["instances"],
            "entities": row["entities"],
            "relations": row["relations"],
            "edges": row["edges"],
        }
        for row in topology["documents"]
    ]
    print_table("Documents", doc_rows)
    print_table("Node types", topology["nodes"])
    print_table("Instance types", topology["instances"])
    print_table("Edge types", topology["edges"])
    print_table("Property types", topology["properties"])


def main() -> int:
    args = parse_args()

    try:
        paths = iter_dclx_files(args.inputs, args.recursive)
        topology = build_topology(paths, args)
        print_topology(topology)
        if args.output:
            args.output.write_text(json.dumps(topology, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"topology.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

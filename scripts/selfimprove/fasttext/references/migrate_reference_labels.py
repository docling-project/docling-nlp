#!/usr/bin/env python3
"""Migrate binary reference CSVs to text, meta-data, and reference labels.

The input directory must contain the compact ``train.csv`` and ``validation.csv``
written by ``selfimprove_references.py``.  The output is a new, independently
resumable self-improvement run; the source run is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from selfimprove_references import (
    LABELS,
    LOG,
    accepted_labels,
    configure_logging,
    confusion_report,
    deterministic_text_fragment,
    digest,
    local_reviews,
    training_label_counts,
    write_json,
)
from tqdm import tqdm

SOURCE_COLUMNS = {"label", "conf", "LLM verified", "iteration", "filename", "xpath", "text"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing the binary train.csv and validation.csv files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory receiving the migrated, resumable three-label run.",
    )
    parser.add_argument(
        "--llm-model", default="qwen3.6:27b",
        help="Installed Ollama model used for former reference rows.",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Base URL of the local Ollama service.",
    )
    parser.add_argument(
        "--ollama-timeout", type=float, default=60.0,
        help="Maximum seconds to wait for one Ollama response.",
    )
    parser.add_argument(
        "--llm-context-window", type=int, default=8192,
        help="Maximum Ollama context tokens per isolated review request.",
    )
    parser.add_argument(
        "--llm-batch-size", type=int, default=32,
        help="Former reference rows classified in each Ollama request.",
    )
    parser.add_argument(
        "--llm-thinking", action="store_true",
        help="Enable Ollama reasoning; disabled by default for rapid labelling.",
    )
    parser.add_argument(
        "--review-limit", type=int, default=-1,
        help="Former reference rows to review; -1 reviews every non-trivial row.",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=-1,
        help="Total source rows to migrate; -1 uses the complete dataset.",
    )
    parser.add_argument(
        "--autotune", action="store_true",
        help="Enable native FastText autotuning for the migrated model.",
    )
    parser.add_argument(
        "--autotune-duration", type=int, default=3600,
        help="FastText autotuning budget in seconds.",
    )
    parser.add_argument(
        "--autotune-modelsize", default="100M",
        help="FastText autotuning model-size target.",
    )
    parser.add_argument(
        "--ngram", type=int, default=0,
        help="FastText word n-gram order; zero disables word n-grams.",
    )
    parser.add_argument(
        "--fasttext-retries", type=int, default=2,
        help="Retries after an incomplete FastText artifact/evaluation failure.",
    )
    parser.add_argument("--debug", action="store_true", help="Show compact Ollama decisions.")
    args = parser.parse_args()
    if (args.review_limit == 0 or args.review_limit < -1
            or args.sample_limit == 0 or args.sample_limit < -1
            or args.llm_context_window < 1 or args.llm_batch_size < 1):
        parser.error(
            "review-limit and sample-limit must be positive or -1; "
            "llm-context-window and llm-batch-size must be positive"
        )
    if args.fasttext_retries < 0:
        parser.error("fasttext-retries cannot be negative")
    return args


def confidence(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def source_iteration(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def load_csv_rows(input_dir: Path) -> list[dict[str, Any]]:
    """Load the compact CSVs while preserving their explicit train split."""

    rows: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for split in ("train", "validation"):
        path = input_dir / f"{split}.csv"
        if not path.is_file():
            raise RuntimeError(f"missing source CSV: {path}")
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not SOURCE_COLUMNS.issubset(reader.fieldnames):
                raise RuntimeError(f"{path} does not have the expected compact CSV columns")
            for source in reader:
                original_label = source["label"]
                if original_label not in {"reference", "text", "meta-data"}:
                    raise RuntimeError(f"unsupported label {original_label!r} in {path}")
                item_text = source["text"].strip()
                if not item_text:
                    continue
                candidate_id = digest(
                    f"csv-migration\0{source['filename']}\0{source['xpath']}\0{item_text}"
                )
                if candidate_id in candidate_ids:
                    raise RuntimeError(f"duplicate candidate in source CSVs: {candidate_id}")
                candidate_ids.add(candidate_id)
                rows.append({
                    "candidate-id": candidate_id,
                    "source-path": source["filename"],
                    "xpath": source["xpath"],
                    "source-kind": "csv-migration",
                    "layout-label": "text",
                    "in_reference_section": False,
                    "numval_only": False,
                    "numval_coverage": 0.0,
                    "introduced_iteration": source_iteration(source["iteration"]),
                    "text": item_text,
                    "training-sample": split == "train",
                    "applied-label": original_label,
                    "confidence": confidence(source["conf"]),
                    "label": original_label,
                    "label-source": "existing",
                })
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    # Retain the source dataset's train/validation ratio in bounded test runs.
    if limit == -1 or limit >= len(rows):
        return rows
    training = [row for row in rows if row["training-sample"]]
    validation = [row for row in rows if not row["training-sample"]]
    validation_count = 0
    if validation and limit > 1:
        validation_count = max(1, round(limit * len(validation) / len(rows)))
        validation_count = min(validation_count, len(validation), limit - 1)
    training_count = min(limit - validation_count, len(training))
    validation_count = min(limit - training_count, len(validation))
    return training[:training_count] + validation[:validation_count]


def migrate_labels(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], int]:
    """Reclassify every former reference unless it is deterministic trivial text."""

    reviewable = []
    deterministic = 0
    for row in rows:
        if deterministic_text_fragment(row["text"]):
            row["label"] = "text"
            row["label-source"] = "deterministic"
            deterministic += 1
        elif row["label"] == "reference":
            reviewable.append(row)
    if args.review_limit != -1:
        reviewable = reviewable[:args.review_limit]
    LOG.info(
        "Classifying %d former reference rows with %s; assigning %d trivial fragments to text without Ollama",
        len(reviewable), args.llm_model, deterministic,
    )
    reviews: dict[str, dict[str, Any]] = {}
    progress = tqdm(
        total=len(reviewable), desc="Ollama migration", unit="sample",
        file=sys.stderr, dynamic_ncols=True,
    )
    for start in range(0, len(reviewable), args.llm_batch_size):
        batch = reviewable[start:start + args.llm_batch_size]
        review_items = [{
            "candidate_id": row["candidate-id"],
            "text": row["text"],
            "prediction_label": row["applied-label"],
            "prediction_confidence": row["confidence"],
            "in_reference_section": False,
        } for row in batch]
        try:
            reviews.update(local_reviews(review_items, args))
        except Exception as exc:  # Keep the old reference label on a transient LLM failure.
            reviews.update({
                item["candidate_id"]: {"accepted": False, "error": str(exc)}
                for item in review_items
            })
        progress.update(len(batch))
    progress.close()
    labels = accepted_labels(reviews)
    for row in rows:
        if row["candidate-id"] in labels:
            row["label"] = labels[row["candidate-id"]]
            row["label-source"] = "llm"
    return reviews, deterministic


def write_migrated_datasets(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    # Write compact CSVs directly; model scoring belongs to self-improvement.
    fields = ["label", "conf", "LLM verified", "iteration", "filename", "xpath", "text"]
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for split, selected in (
        ("train", [row for row in rows if row["training-sample"]]),
        ("validation", [row for row in rows if not row["training-sample"]]),
    ):
        with (data_dir / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow({
                    "label": row["label"],
                    "conf": row["confidence"],
                    "LLM verified": "yes" if row["label-source"] == "llm" else "no",
                    "iteration": row["introduced_iteration"],
                    "filename": row["source-path"],
                    "xpath": row["xpath"],
                    "text": row["text"],
                })


def candidate_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert migrated training rows to the candidates required by --resume."""

    return [{
        "candidate_id": row["candidate-id"],
        "source_path": row["source-path"],
        "source_kind": row["source-kind"],
        "xpath": row["xpath"],
        "layout_label": row["layout-label"],
        "in_reference_section": row["in_reference_section"],
        "numval_only": False,
        "numval_coverage": 0.0,
        "numval_error": None,
        "introduced_iteration": row["introduced_iteration"],
        "text": row["text"],
        "prediction_label": row["label"],
        "prediction_confidence": row["confidence"],
        "error": None,
    } for row in rows]


def main() -> int:
    args = arguments()
    configure_logging(args.debug)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_csv_rows(args.input_dir)
    rows = limit_rows(source_rows, args.sample_limit)
    LOG.info(
        "Loaded %d of %d source data rows from %s",
        len(rows), len(source_rows), args.input_dir,
    )
    reviews, deterministic = migrate_labels(rows, args)
    counts = Counter(row["label"] for row in rows)
    LOG.info("Migrated label counts: %s", dict(sorted(counts.items())))

    migration_iteration = max(row["introduced_iteration"] for row in rows)
    validation_rows = [row for row in rows if not row["training-sample"]]
    review_report = confusion_report([
        ("reference", label) for label in accepted_labels(reviews).values()
    ])
    candidate_rows = candidate_artifacts(rows)
    labels = {row["candidate-id"]: row["label"] for row in rows}
    llm_labels = accepted_labels(reviews)
    summary = {
        "iteration": migration_iteration,
        "baseline": "binary-csv-migration",
        "candidates": len(candidate_rows),
        "reviewed": len(llm_labels),
        "deterministic_text": deterministic,
        "labels": dict(sorted(counts.items())),
        "validation_labels": {
            label: sum(row["label"] == label for row in validation_rows)
            for label in sorted(LABELS)
        },
        "training_label_counts": training_label_counts(rows),
    }

    write_migrated_datasets(args.output_dir, rows)
    write_json(args.output_dir / "reference-candidates.json", candidate_rows)
    write_json(args.output_dir / "semantic-predictions.json", candidate_rows)
    write_json(args.output_dir / "bootstrap-labels.json", labels)
    write_json(args.output_dir / "llm-corrections.json", llm_labels)
    reviews_dir = args.output_dir / "metadata" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    write_json(reviews_dir / f"migration-{migration_iteration:03d}.json", [
        {"candidate-id": candidate_id, **review}
        for candidate_id, review in reviews.items()
    ])
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    write_json(metrics_dir / "migration-review-confusion.json", review_report)
    write_json(args.output_dir / "migration-summary.json", summary)
    LOG.info(
        "Migration complete; bootstrap and train with --output-dir %s --resume",
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

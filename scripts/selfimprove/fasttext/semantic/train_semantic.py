#!/usr/bin/env python3
"""Download, prepare, train, and evaluate Docling's semantic classifier."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from docling_nlp.andromeda_nlp import nlp_model
from huggingface_hub import hf_hub_download

LOG = logging.getLogger(__name__)
LABELS = {"meta-data", "reference", "text"}
OMITTED_LABELS = {"header"}
REQUIRED_COLUMNS = {"label", "text"}
LOSS_VALUES = {"softmax", "ova", "one-vs-all", "hs", "ns"}
TRAINING_ARGUMENTS = {
    "learning_rate": "learning-rate",
    "epochs": "epoch",
    "dimension": "dim",
    "context_window": "ws",
    "ngram": "n-gram",
    "loss": "loss",
    "min_count": "min-count",
    "min_char_ngram": "min-char-ngram",
    "max_char_ngram": "max-char-ngram",
    "bucket": "bucket",
    "threads": "thread",
    "seed": "seed",
}


def configured_path(value: str) -> Path:
    """Resolve configuration paths relative to the current working directory."""

    return Path(value).expanduser().resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")
    for section in ("data", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration mapping: {section}")

    for key in ("repo_id", "train_file", "validation_file"):
        if not config["data"].get(key):
            raise ValueError(f"missing configuration value: data.{key}")
    training = config["training"]
    duration = training.get("duration", 3600)
    if not isinstance(duration, int) or duration < 1:
        raise ValueError("training.duration must be a positive integer")
    predictions = training.get("predictions", 1)
    if not isinstance(predictions, int) or predictions < 1:
        raise ValueError("training.predictions must be a positive integer")
    if not isinstance(training.get("autotune", True), bool):
        raise ValueError("training.autotune must be true or false")
    if not isinstance(training.get("metric", "f1"), str):
        raise ValueError("training.metric must be a string")

    positive_integers = ("epochs", "dimension", "context_window", "threads")
    non_negative_integers = (
        "ngram",
        "min_count",
        "min_char_ngram",
        "max_char_ngram",
        "bucket",
    )
    for key in positive_integers:
        value = training.get(key)
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f"training.{key} must be null or a positive integer")
    for key in non_negative_integers:
        value = training.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError(f"training.{key} must be null or a non-negative integer")
    seed = training.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("training.seed must be null or an integer")
    learning_rate = training.get("learning_rate")
    if learning_rate is not None and (
        not isinstance(learning_rate, (int, float)) or learning_rate <= 0
    ):
        raise ValueError("training.learning_rate must be null or positive")
    loss = training.get("loss")
    if loss is not None and loss not in LOSS_VALUES:
        raise ValueError(
            "training.loss must be null or one of: " + ", ".join(sorted(LOSS_VALUES))
        )
    minn = training.get("min_char_ngram")
    maxn = training.get("max_char_ngram")
    if minn is not None and maxn is not None and minn > maxn:
        raise ValueError("training.min_char_ngram cannot exceed max_char_ngram")
    return config


def download_file(
    repo_id: str,
    filename: str,
    revision: str,
    destination: Path,
    force: bool = False,
) -> Path:
    """Download a dataset file only when the local copy is absent."""

    if destination.is_file() and not force:
        LOG.info("Reusing dataset file: %s", destination)
        return destination

    LOG.info("Downloading %s from %s at %s", filename, repo_id, revision)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        force_download=force,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)
    return destination


def convert_csv(
    source: Path,
    stream: Any,
    training_sample: bool,
) -> tuple[Counter[str], int]:
    """Append one CSV split in the JSONL format consumed by native training."""

    counts: Counter[str] = Counter()
    omitted = 0
    with source.open(encoding="utf-8", newline="") as csv_stream:
        reader = csv.DictReader(csv_stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{source} is missing columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            text = row.get("text") or ""
            if label in OMITTED_LABELS:
                omitted += 1
                continue
            if label not in LABELS:
                raise ValueError(f"unknown label {label!r} at {source}:{line_number}")
            if not text.strip():
                raise ValueError(f"empty text at {source}:{line_number}")

            record = {
                "label": label,
                "text": text,
                "training-sample": training_sample,
                "source-file": row.get("filename"),
                "source-path": row.get("xpath"),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[label] += 1
    if not counts:
        raise ValueError(f"dataset split has no usable records: {source}")
    return counts, omitted


def prepare_dataset(
    train_csv: Path, validation_csv: Path, output: Path
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        train_counts, omitted_train = convert_csv(train_csv, stream, True)
        validation_counts, omitted_validation = convert_csv(
            validation_csv, stream, False
        )
    return {
        "training": dict(sorted(train_counts.items())),
        "validation": dict(sorted(validation_counts.items())),
        "omitted-header": omitted_train + omitted_validation,
    }


def semantic_config(
    dataset: Path,
    model_file: Path,
    metrics_file: Path,
    training: dict[str, Any],
) -> dict[str, Any]:
    args = {
        native_name: training[yaml_name]
        for yaml_name, native_name in TRAINING_ARGUMENTS.items()
        if training.get(yaml_name) is not None
    }
    return {
        "mode": "train",
        "model": "semantic",
        "hpo": {
            "autotune": bool(training.get("autotune", True)),
            "duration": training.get("duration", 3600),
            "modelsize": training.get("model_size", "100M"),
            "metric": training.get("metric", "f1"),
            "predictions": training.get("predictions", 1),
        },
        "args": args,
        "files": {
            "data-file": str(dataset),
            "model-file": str(model_file),
            "metrics-file": str(metrics_file),
        },
    }


def ensure_success(result: dict[str, Any], operation: str) -> None:
    status = result.get("model-training", {})
    if not status.get("success"):
        raise RuntimeError(status.get("message", f"semantic {operation} failed"))


def train(config: dict[str, Any], force_download: bool = False) -> dict[str, Path]:
    output = config["output"]
    output_dir = configured_path(output.get("directory", "artifacts/semantic"))
    data_dir = output_dir / output.get("data_directory", "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = config["data"]
    repo_id = data["repo_id"]
    revision = str(data.get("revision", "main"))
    train_csv = download_file(
        repo_id,
        data["train_file"],
        revision,
        data_dir / Path(data["train_file"]).name,
        force_download,
    )
    validation_csv = download_file(
        repo_id,
        data["validation_file"],
        revision,
        data_dir / Path(data["validation_file"]).name,
        force_download,
    )

    dataset = output_dir / output.get("dataset_file", "semantic.jsonl")
    model_file = output_dir / output.get("model_file", "fst_semantic.bin")
    metrics_file = output_dir / output.get("metrics_file", "validation_metrics.txt")
    manifest_file = output_dir / output.get("manifest_file", "manifest.json")
    counts = prepare_dataset(train_csv, validation_csv, dataset)
    LOG.info("Prepared semantic labels: %s", counts)

    native_config = semantic_config(
        dataset, model_file, metrics_file, config["training"]
    )
    model = nlp_model()
    model.set_loglevel(config["training"].get("loglevel", "WARNING"))
    ensure_success(model.prepare_data_for_train(native_config), "preparation")
    ensure_success(model.train(native_config), "training")
    ensure_success(model.evaluate(native_config), "evaluation")

    expected = [
        model_file,
        model_file.with_suffix(".vec"),
        metrics_file,
        Path(f"{dataset}.fasttext.train.txt"),
        Path(f"{dataset}.fasttext.validate.txt"),
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("training omitted expected artifacts: " + ", ".join(missing))

    manifest = {
        "dataset": {"repo-id": repo_id, "revision": revision},
        "labels": sorted(LABELS),
        "counts": counts,
        "training": config["training"],
        "artifacts": {path.name: str(path) for path in expected},
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "dataset": dataset,
        "model": model_file,
        "metrics": metrics_file,
        "manifest": manifest_file,
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace locally downloaded dataset CSVs.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    artifacts = train(load_config(args.config), args.force_download)
    for name, path in artifacts.items():
        LOG.info("%s: %s", name.capitalize(), path)
    print(artifacts["metrics"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
